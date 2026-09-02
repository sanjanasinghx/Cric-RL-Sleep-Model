"""Print seed 4 in the Table 3 format, next to the published seeds.

Formatting is imported from verify_tables.py. No group statistic is recomputed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))

from verify_tables import HEADER, WIDTHS, _row, q  # noqa: E402

SEED4 = OUT / "results" / "metrics_seed4.json"
PUBLISHED = ROOT / "results" / "metrics_aggregated.json"


def row_for(m: dict) -> str:
    c = m["trained_action_counts"]
    return _row([
        m["seed"],
        f"{m['episodes_trained']:,}",
        f"{q(m['baseline_pure_awake_return'], 2)}",
        f"{q(m['trained_eval_return'], 2):+}",
        m["trained_eval_length"],
        m["trained_eval_terminal_cause"],
        f"{q(m['tail_mean_return'], 2)}",
        f"{q(m['mean_overload'], 3)}",
        f"{q(m['mean_recall'], 3)}",
        f"{c['AWAKE']} / {c['NREM']} / {c['REM']}",
    ])


def main() -> None:
    m = json.loads(SEED4.read_text())["per_seed"][0]
    line = "-" * (sum(WIDTHS) + 2 * (len(WIDTHS) - 1))

    print("Seed 4 (not part of the paper sweep)")
    print(f"Read from {SEED4}\n")
    print(_row(HEADER))
    print(line)
    print(row_for(m))
    print(line)

    print("\nPublished seeds (results/metrics_aggregated.json):\n")
    print(_row(HEADER))
    print(line)
    for pm in json.loads(PUBLISHED.read_text())["per_seed"]:
        print(row_for(pm))
    print(line)

    tail, base = m["tail_mean_return"], m["baseline_pure_awake_return"]
    c = m["trained_action_counts"]
    print(f"\ntail mean {tail:.3f} vs baseline {base:.3f} (margin {tail - base:+.3f})")
    print(f"actions: AWAKE {c['AWAKE']}, NREM {c['NREM']}, REM {c['REM']}")
    print(f"terminal: {m['trained_eval_terminal_cause']} at step {m['trained_eval_length']} of 300")


if __name__ == "__main__":
    main()
