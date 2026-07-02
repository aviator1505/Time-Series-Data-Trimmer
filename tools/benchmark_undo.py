"""Benchmark undo-stack cost: snapshot time and memory per operation.

Usage: python tools/benchmark_undo.py [rows] [cols]

Measures a representative interactive session (mark_bad, annotate,
filter apply, segment deletion) on a synthetic dataset and reports the
time spent per operation and the undo-stack memory footprint.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsdt_core.core import CoreDataModel


def build_df(rows: int, cols: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    data = {"normalized_time": np.arange(rows) / 120.0}
    for i in range(cols - 1):
        data[f"ch_{i}"] = rng.standard_normal(rows)
    return pd.DataFrame(data)


def main() -> None:
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    df = build_df(rows, cols)
    m = CoreDataModel()
    m.load_frame(df, "bench.csv")
    t_max = float(m.df["normalized_time"].max())

    def timed(label, fn):
        t0 = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t0) * 1000
        print(f"{label:<28s} {dt:9.1f} ms   undo stack: {m._estimate_undo_memory_mb():8.1f} MB "
              f"({len(m._undo_stack)} entries)")

    timed("annotate", lambda: m.annotate(1.0, 2.0, "bench"))
    timed("mark_bad", lambda: m.mark_bad(3.0, 4.0))
    timed("update_annotation", lambda: m.update_annotation(
        m.annotations[0].id, 1.0, 2.5, "bench", None, None))

    filtered = m.get_dataframe()
    filtered["ch_0"] = filtered["ch_0"].rolling(5, min_periods=1).mean()
    timed("apply_dataframe (1 chan)", lambda: m.apply_dataframe(
        filtered, "filter", 0.0, t_max, {"channels": ["ch_0"], "filter_type": "moving_average"}))

    timed("delete_segment (1% rows)", lambda: m.delete_segment(t_max * 0.4, t_max * 0.41))
    timed("undo x3", lambda: [m.undo() for _ in range(3)])
    timed("redo x3", lambda: [m.redo() for _ in range(3)])

    print(f"\ndataset: {rows:,} rows x {cols} cols "
          f"({df.memory_usage(deep=True).sum() / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
