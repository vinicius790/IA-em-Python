"""Regras puras da 8.1: perfil de CSV, canário, papéis e intervalo de confiança.

Os módulos testados aqui usam só a biblioteca padrão, então este arquivo também roda
sem nenhuma dependência instalada: `python -m unittest tests.test_stdlib_units`.
"""

import unittest

from app.data_catalog.csv_profile import infer_target, profile_csv, training_problems, valid_dataset_name
from app.deployment.canary import (
    PROMOTE,
    ROLLBACK,
    WAIT,
    CanaryRule,
    arm_stats,
    choose_arm,
    decide,
    validate_canary_attribution,
)
from app.governance.rbac import can, generate_key, hash_key, required_role, same_key
from app.modeling.intervals import confidence_interval, t_critical

WINE = "acidez,acucar,teor,origem,qualidade\n" + "".join(
    f"{5 + i % 7 * 0.3:.1f},{1 + i % 5},{9 + i % 4 * 0.5},"
    f"{'sul' if i % 3 else 'norte'},{'boa' if i % 2 else 'ruim'}\n"
    for i in range(40)
)


class CsvProfileTest(unittest.TestCase):
    def test_types_statistics_and_inferred_target(self):
        profile = profile_csv(WINE)
        self.assertEqual((profile["rows"], profile["columns"]), (40, 5))
        self.assertEqual(profile["target"], "qualidade")
        self.assertTrue(profile["target_inferred"])
        self.assertEqual(profile["features"], ["acidez", "acucar", "teor"])
        self.assertEqual(profile["classes"], {"ruim": 20, "boa": 20})
        columns = {column["name"]: column for column in profile["column_profiles"]}
        self.assertEqual(columns["origem"]["semantic_type"], "categorical")
        self.assertEqual(columns["origem"]["top_values"], {"sul": 26, "norte": 14})
        acucar = columns["acucar"]["numeric"]
        self.assertEqual((acucar["min"], acucar["max"], acucar["mean"], acucar["p50"]), (1.0, 5.0, 3.0, 3.0))
        self.assertAlmostEqual(acucar["std"], 2**0.5, places=6)
        self.assertEqual(training_problems(profile), [])

    def test_nulls_are_counted_and_do_not_break_numeric_inference(self):
        profile = profile_csv("a,b\n1,x\n,y\nNA,x\n4,?\n")
        a, b = profile["column_profiles"]
        self.assertEqual((a["semantic_type"], a["nulls"], a["null_rate"]), ("numeric", 2, 0.5))
        self.assertEqual(a["numeric"]["mean"], 2.5)
        self.assertEqual((b["nulls"], b["unique"]), (1, 2))

    def test_explicit_target_wins_and_integer_classes_are_accepted(self):
        text = "x,y,grupo\n" + "".join(f"{i},{i * 2},{i % 3}\n" for i in range(30))
        self.assertEqual(profile_csv(text)["target"], "grupo")
        chosen = profile_csv(text, target="y")
        self.assertEqual((chosen["target"], chosen["target_inferred"]), ("y", False))
        self.assertEqual(chosen["features"], ["x", "grupo"])

    def test_continuous_last_column_is_not_a_target(self):
        columns = profile_csv("a,b\n" + "".join(f"{i},{i * 1.5}\n" for i in range(40)))
        self.assertIsNone(columns["target"])
        self.assertIn("no target column", training_problems(columns)[0])
        self.assertIsNone(infer_target([], 0))

    def test_duplicates_and_bom(self):
        profile = profile_csv("﻿a,b\n1,x\n1,x\n2,y\n")
        self.assertEqual(profile["duplicate_rows"], 1)
        self.assertEqual(profile["column_profiles"][0]["name"], "a")

    def test_invalid_files_are_rejected_with_a_reason(self):
        for text, reason in [
            ("", "empty"),
            ("a,b\n", "no data rows"),
            ("a,a\n1,2\n", "unique"),
            ("a,,c\n1,2,3\n", "unique"),
            ("a,b\n1,2,3\n", "line 2 has 3 fields"),
        ]:
            with self.assertRaisesRegex(ValueError, reason):
                profile_csv(text)
        with self.assertRaisesRegex(ValueError, "Target column 'z' not found"):
            profile_csv("a,b\n1,2\n", target="z")
        # `csv.Error` (aqui, campo acima do limite do módulo csv) também vira ValueError, não 500.
        with self.assertRaisesRegex(ValueError, "Malformed CSV"):
            profile_csv("a,b\n" + "x" * 200_000 + ",1\n")

    def test_only_plain_decimal_numbers_are_numeric(self):
        # `float()` aceitaria todos estes, mas o treino (pandas) os leria como texto.
        for value in ("1_000", "inf", "-Infinity", "١٢", "0x10", "1e999"):
            column = profile_csv(f"a,b\n{value},x\n2,y\n")["column_profiles"][0]
            self.assertEqual(column["semantic_type"], "categorical", value)
        column = profile_csv("a,b\n+1.5,x\n.5,y\n 2e3 ,z\n-3.,w\n")["column_profiles"][0]
        self.assertEqual((column["semantic_type"], column["numeric"]["max"]), ("numeric", 2000.0))

    def test_training_problems_name_every_blocker(self):
        small = profile_csv("texto,classe\nfoo,a\nbar,b\n")
        problems = training_problems(small)
        self.assertTrue(any("30 rows" in item for item in problems))
        self.assertTrue(any("two rows" in item for item in problems))
        self.assertTrue(any("numeric feature" in item for item in problems))
        one_class = profile_csv("x,c\n" + "1,a\n" * 30, target="c")
        self.assertIn("target must have at least two classes", training_problems(one_class))
        with_nulls = profile_csv("x,c\n" + "1,a\n2,b\n" * 20 + "3,\n", target="c")
        self.assertIn("target column has missing values", training_problems(with_nulls))

    def test_training_problems_mirror_the_training_quality_gates(self):
        # O gate do treino recusa nulos nas features e mais de 1% de linhas repetidas
        # (features + alvo); o upload recusa os mesmos casos, com o motivo.
        unique = "".join(f"{i},{i % 2},texto{i % 3},{'ab'[i % 2]}\n" for i in range(40))
        self.assertEqual(training_problems(profile_csv("x,y,t,c\n" + unique)), [])
        missing = profile_csv("x,y,t,c\n" + unique + ",1,texto,a\n")
        self.assertIn("feature columns have missing values: x", training_problems(missing))
        repeated = profile_csv("x,y,t,c\n" + unique + "1,1,outro,b\n")
        self.assertEqual(repeated["duplicate_training_rows"], 1)
        self.assertEqual(repeated["duplicate_rows"], 0)  # a coluna de texto difere, mas não entra no treino
        self.assertTrue(any("repeat the same features" in item for item in training_problems(repeated)))
        # "1.0" e "1" são o mesmo número para o treino.
        same_number = profile_csv("x,y,t,c\n" + unique + "1.0,1,texto1,b\n")
        self.assertEqual(same_number["duplicate_training_rows"], 1)

    def test_dataset_names_are_safe_for_paths(self):
        for name in ("vinhos", "vinhos-2026", "a_b", "9lives"):
            self.assertTrue(valid_dataset_name(name), name)
        for name in ("", "Vinhos", "../etc", "a/b", "a.b", "-x", "x" * 65, "nome com espaço"):
            self.assertFalse(valid_dataset_name(name), name)


def _rows(n, confidence=0.9, latency=2.0, correct=None):
    """`n` predições; `correct` (0..1) rotula todas e acerta essa fração."""
    rows = []
    for i in range(n):
        actual = None
        if correct is not None:
            actual = "a" if i < round(n * correct) else "b"
        rows.append(
            {"confidence": confidence, "latency_ms": latency, "predicted_class": "a", "actual_class": actual}
        )
    return rows


class CanaryTest(unittest.TestCase):
    def test_prediction_attribution_is_all_or_nothing_and_accepts_only_known_arms(self):
        self.assertEqual(validate_canary_attribution(None, None), (None, None))
        self.assertEqual(validate_canary_attribution(7, "stable"), (7, "stable"))
        self.assertEqual(validate_canary_attribution(7, "candidate"), (7, "candidate"))
        for canary_id, arm in ((None, "stable"), (7, None), (0, "candidate"), (True, "stable"), (7, "other")):
            with self.subTest(canary_id=canary_id, arm=arm), self.assertRaises(ValueError):
                validate_canary_attribution(canary_id, arm)

    def test_traffic_split_is_deterministic_and_close_to_the_percentage(self):
        keys = [f"req-{i}" for i in range(5000)]
        share = sum(choose_arm(key, 20) == "candidate" for key in keys) / len(keys)
        self.assertAlmostEqual(share, 0.20, delta=0.03)
        self.assertEqual([choose_arm(k, 20) for k in keys[:50]], [choose_arm(k, 20) for k in keys[:50]])
        self.assertTrue(all(choose_arm(key, 0) == "stable" for key in keys[:200]))
        self.assertTrue(all(choose_arm(key, 100) == "candidate" for key in keys[:200]))
        # Aumentar o percentual só move chaves da estável para a candidata, nunca o contrário.
        moved = [key for key in keys if choose_arm(key, 20) == "candidate"]
        self.assertTrue(all(choose_arm(key, 50) == "candidate" for key in moved))
        with self.assertRaises(ValueError):
            choose_arm("x", 101)

    def test_arm_stats(self):
        self.assertEqual(
            arm_stats([]),
            {"requests": 0, "mean_confidence": None, "p95_latency_ms": None, "labeled": 0, "accuracy": None},
        )
        rows = _rows(20, confidence=0.8, latency=1.0, correct=0.75)
        rows[0]["latency_ms"] = 50.0
        stats = arm_stats(rows)
        self.assertEqual((stats["requests"], stats["labeled"], stats["accuracy"]), (20, 20, 0.75))
        self.assertEqual((stats["mean_confidence"], stats["p95_latency_ms"]), (0.8, 1.0))
        self.assertEqual(arm_stats(rows[:2])["p95_latency_ms"], 50.0)

    def test_waits_until_the_candidate_has_enough_requests(self):
        rule = CanaryRule(min_requests=30)
        result = decide(arm_stats(_rows(100)), arm_stats(_rows(29)), rule)
        self.assertEqual(result["decision"], WAIT)
        self.assertIn("29 of 30", result["reason"])

    def test_promotes_a_candidate_that_meets_every_check(self):
        rule = CanaryRule(min_requests=30, min_labeled=10)
        stable = arm_stats(_rows(100, confidence=0.90, latency=2.0, correct=0.9))
        candidate = arm_stats(_rows(40, confidence=0.88, latency=2.5, correct=0.9))
        result = decide(stable, candidate, rule)
        self.assertEqual(result["decision"], PROMOTE)
        self.assertEqual([check["name"] for check in result["checks"]], ["confidence", "latency", "accuracy"])
        self.assertTrue(all(check["passed"] for check in result["checks"]))

    def test_rolls_back_on_each_kind_of_regression(self):
        rule = CanaryRule(min_requests=30, min_labeled=10)
        stable = arm_stats(_rows(100, confidence=0.90, latency=2.0, correct=0.9))
        for candidate_rows, failed in [
            (_rows(40, confidence=0.70, latency=2.0, correct=0.9), "confidence"),
            (_rows(40, confidence=0.90, latency=9.0, correct=0.9), "latency"),
            (_rows(40, confidence=0.90, latency=2.0, correct=0.5), "accuracy"),
        ]:
            result = decide(stable, arm_stats(candidate_rows), rule)
            self.assertEqual(result["decision"], ROLLBACK, failed)
            self.assertEqual(result["reason"], f"candidate failed: {failed}")

    def test_accuracy_is_ignored_without_enough_labels_and_idle_stable_does_not_block(self):
        rule = CanaryRule(min_requests=30, min_labeled=50)
        stable = arm_stats(_rows(100, correct=0.9))
        candidate = arm_stats(_rows(40, correct=0.1))
        result = decide(stable, candidate, rule)
        self.assertEqual(result["decision"], PROMOTE)
        self.assertNotIn("accuracy", [check["name"] for check in result["checks"]])
        idle = decide(arm_stats([]), arm_stats(_rows(40)), rule)
        self.assertEqual((idle["decision"], idle["checks"]), (PROMOTE, []))

    def test_rule_validates_its_limits(self):
        # Acima da janela de métricas de cada braço (500), a regra esperaria para sempre.
        for kwargs in (
            {"min_requests": 0},
            {"min_requests": 501},
            {"min_labeled": 501},
            {"max_latency_ratio": 0.5},
            {"max_confidence_drop": 2},
        ):
            with self.assertRaises(ValueError):
                CanaryRule(**kwargs)
        self.assertEqual(CanaryRule(**CanaryRule(min_requests=7).as_dict()).min_requests, 7)


class RbacTest(unittest.TestCase):
    def test_role_hierarchy(self):
        self.assertTrue(can("admin", "admin") and can("admin", "reader") and can("operator", "reader"))
        self.assertFalse(can("reader", "operator") or can("operator", "admin"))
        self.assertFalse(can("desconhecido", "reader"))
        with self.assertRaises(ValueError):
            can("admin", "root")

    def test_required_role_by_method_and_path(self):
        self.assertEqual(required_role("GET", "/api/v1/models"), "reader")
        self.assertEqual(required_role("POST", "/api/v1/predict"), "reader")
        self.assertEqual(required_role("POST", "/api/v1/predict/batch"), "reader")
        self.assertEqual(required_role("POST", "/api/v8/predict"), "reader")
        self.assertEqual(required_role("POST", "/api/v2/explain"), "reader")
        self.assertEqual(required_role("POST", "/api/v1/training/train"), "operator")
        self.assertEqual(required_role("POST", "/api/v2/predictions/7/feedback"), "operator")
        self.assertEqual(required_role("DELETE", "/api/v1/iris/1"), "operator")
        self.assertEqual(required_role("POST", "/api/v8/datasets/upload"), "operator")

    def test_keys_are_random_and_only_the_hash_is_comparable(self):
        first, second = generate_key(), generate_key()
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("iris_") and len(first) > 40)
        self.assertEqual(len(hash_key(first)), 64)
        self.assertNotIn(first, hash_key(first))
        self.assertTrue(same_key(first, first))
        self.assertFalse(same_key(first, second))


class IntervalTest(unittest.TestCase):
    def test_t_interval_matches_a_known_value(self):
        # Média 0.94, desvio amostral 0.0158114, n=5, t(4; 95%) = 2.776 -> margem 0.019629.
        interval = confidence_interval([0.92, 0.93, 0.94, 0.95, 0.96])
        self.assertEqual((interval["mean"], interval["n"], interval["confidence"]), (0.94, 5, 0.95))
        self.assertAlmostEqual(interval["std"], 0.015811, places=6)
        self.assertAlmostEqual(interval["lower"], 0.920371, places=5)
        self.assertAlmostEqual(interval["upper"], 0.959629, places=5)

    def test_wider_confidence_gives_a_wider_interval_and_constant_scores_collapse(self):
        scores = [0.90, 0.95, 0.85, 0.92]
        widths = [
            confidence_interval(scores, level)["upper"] - confidence_interval(scores, level)["lower"]
            for level in (0.90, 0.95, 0.99)
        ]
        self.assertEqual(widths, sorted(widths))
        flat = confidence_interval([1.0, 1.0, 1.0])
        self.assertEqual((flat["lower"], flat["upper"], flat["std"]), (1.0, 1.0, 0.0))

    def test_critical_values_and_invalid_input(self):
        self.assertEqual(t_critical(1), 12.706)
        self.assertEqual(t_critical(19, 0.99), 2.861)
        self.assertAlmostEqual(t_critical(500), 1.96, places=2)
        with self.assertRaisesRegex(ValueError, "confidence must be one of"):
            t_critical(4, 0.8)
        with self.assertRaisesRegex(ValueError, "two observations"):
            confidence_interval([0.9])


if __name__ == "__main__":
    unittest.main()
