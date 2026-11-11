
"""Tune and benchmark one matmul with TVM MetaSchedule, then save the results."""
import json
from pathlib import Path
import time

import numpy as np
import tvm
from tvm import te
from tvm.s_tir import meta_schedule as ms

# ---------------------------------------------------------------
# Settings. Only TARGET and RUNNER change when you move to the Pi.
# ---------------------------------------------------------------
N = 512                                                      # matrix size (N x N)
TARGET = tvm.target.Target({"kind": "llvm", "num-cores": 4}) # Mac CPU for now
RUNNER = "local"                                             # later: an RPC runner pointing at the Pi
STRATEGY = "evolutionary"                                    # search strategy under test (replay-trace for random search)
MAX_TRIALS = 64                                              # total trial budget
TRIALS_PER_ITER = 16                                         # 64 / 16 = 4 rounds of search
SEED = 0
REPO_DIR = Path(__file__).resolve().parent.parent
WORK_DIR = REPO_DIR / f"logs/raw/{STRATEGY}_n{N}_t{MAX_TRIALS}_s{SEED}"
RESULTS_DIR = REPO_DIR / "results"
RESULTS_FILENAME = f"{STRATEGY}_n{N}_t{MAX_TRIALS}_s{SEED}.json"
REPEATS = 20                                                 # number of timing samples; result uses their median
NUMBER = 10                                                  # runs averaged within each timing sample


def make_matmul(n):
    """Return a TIR PrimFunc computing C = A @ B for n x n float32 matrices."""

    A = te.placeholder((n, n), dtype="float32", name="A")
    B = te.placeholder((n, n), dtype="float32", name="B")
    k = te.reduce_axis((0, n), name="k")
    C = te.compute((n, n), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    return te.create_prim_func([A, B, C])


def time_module(lib, n):
    """Return the median runtime in milliseconds of a compiled matmul module."""

    dev = tvm.cpu()
    to_dev = getattr(tvm.runtime, "tensor", None) or tvm.nd.array
    rng = np.random.default_rng(SEED)
    a_np = rng.random((n, n), dtype=np.float32)
    b_np = rng.random((n, n), dtype=np.float32)
    a, b = to_dev(a_np, dev), to_dev(b_np, dev)
    c = to_dev(np.zeros((n, n), dtype="float32"), dev)

    mod = lib.jit()
    entry = mod["main"]
    # Run correctness once before timing; this also warms the entry point.
    entry(a, b, c)
    np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-4, atol=1e-4)

    result = mod.time_evaluator("main", dev, repeat=REPEATS, number=NUMBER)(a, b, c)
    return result.median * 1000.0

def time_untuned(func):
    """Compile with no tuning and return its runtime (ms). This is your floor."""

    lib = tvm.compile(func, target=TARGET)
    return time_module(lib, N)


def tune(func):
    """Run MetaSchedule with the configured search strategy and return the database."""

    return ms.tune_tir(
        mod=func,
        target=TARGET,
        work_dir=str(WORK_DIR),
        max_trials_global=MAX_TRIALS,
        num_trials_per_iter=TRIALS_PER_ITER,
        runner=RUNNER,
        seed=SEED,
        space="post-order-apply",
        cost_model="xgb",
        strategy=STRATEGY,
        builder="local",
        database="json",
    )


def time_tuned(database, func):
    """Compile and return the runtime (ms) of the best schedule found."""
    
    sch = ms.tir_integration.compile_tir(database, func, TARGET)
    if sch is None:
        raise RuntimeError(f"Tuning found no valid schedule; check the logs in {WORK_DIR}")
    lib = tvm.compile(sch.mod, target=TARGET)
    return time_module(lib, N)


def save_results(results):
    """Write settings and measured runtimes to the tracked results directory.

    ``results`` should contain the untuned_ms, tuned_ms, and speedup values.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / RESULTS_FILENAME

    data = {
        "settings": {
            "N": N,
            "TARGET": json.loads(str(TARGET)),
            "STRATEGY": STRATEGY,
            "TVM_VERSION": tvm.__version__,
            "MAX_TRIALS": MAX_TRIALS,
            "TRIALS_PER_ITER": TRIALS_PER_ITER,
            "SEED": SEED,
            "REPEATS": REPEATS,
            "NUMBER": NUMBER,
        },
        **results,
    }
    with output_path.open("w", encoding="utf-8") as result_file:
        json.dump(data, result_file, indent=2)
        result_file.write("\n")
    return output_path


if __name__ == "__main__":  # required on macOS: the local runner spawns processes
    if WORK_DIR.exists():
        raise SystemExit(
            f"Tuning directory already exists: {WORK_DIR}. "
            "Delete it or change the settings before running again."
        )

    func = make_matmul(N)
    untuned_ms = time_untuned(func)
    tuning_start = time.perf_counter()
    database = tune(func)
    tuning_wall_time_seconds = time.perf_counter() - tuning_start
    tuned_ms = time_tuned(database, func)
    speedup = untuned_ms / tuned_ms

    print(f"untuned {untuned_ms:.3f} ms -> tuned {tuned_ms:.3f} ms ({speedup:.2f}x speedup)")
    results_path = save_results({
        "untuned_ms": untuned_ms,
        "tuned_ms": tuned_ms,
        "speedup": speedup,
        "tuning_wall_time_seconds": tuning_wall_time_seconds,
    })
    print(f"Results saved to {results_path}")
