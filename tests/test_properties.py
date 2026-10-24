import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from app.monitoring.drift import ks_test, psi

samples = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=5, max_size=200
)


@settings(max_examples=200, deadline=None)
@given(samples)
def test_psi_is_zero_for_identical_samples(values):
    assert psi(values, values) == 0.0


@settings(max_examples=200, deadline=None)
@given(samples, samples)
def test_psi_is_non_negative(ref, cur):
    assert psi(ref, cur) >= 0.0


@settings(max_examples=200, deadline=None)
@given(samples)
def test_ks_identical_samples_have_no_drift(values):
    res = ks_test(values, values)
    assert res["statistic"] == 0.0
    assert not res["drift_detected"]


@settings(max_examples=200, deadline=None)
@given(samples, samples)
def test_ks_statistic_is_symmetric_and_bounded(a, b):
    ab, ba = ks_test(a, b), ks_test(b, a)
    assert ab["statistic"] == ba["statistic"]
    assert 0.0 <= ab["statistic"] <= 1.0
    assert 0.0 <= ab["p_value"] <= 1.0


@settings(max_examples=50, deadline=None)
@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_large_shift_is_detected(seed):
    rng = np.random.default_rng(seed)
    ref = rng.normal(0, 1, 500)
    assert ks_test(ref, ref + 5)["drift_detected"]
    assert psi(ref, ref + 5) >= 0.25
