import time


def check(name, fn):
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    print(f"{name}: {t1 - t0:.2f}s", flush=True)


print("Starting checks...", flush=True)
check("pandas", lambda: __import__("pandas"))
check("fastapi", lambda: __import__("fastapi"))
check("sklearn", lambda: __import__("sklearn"))
check("xgboost", lambda: __import__("xgboost"))
check("app.config", lambda: __import__("app.config"))
check("app.database", lambda: __import__("app.database"))
check("app.ml", lambda: __import__("app.ml"))
check("app.main", lambda: __import__("app.main"))
print("Done all checks!", flush=True)
