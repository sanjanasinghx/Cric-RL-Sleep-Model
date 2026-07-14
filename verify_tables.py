"""Print paper Table 3 straight from results/metrics_aggregated.json.

Every value is read from the aggregated metrics file. Nothing is recomputed and
nothing is hard coded, so the printed rows are exactly the committed sweep result.
Rounding is half-up on the stored value, which is how the paper table was rounded.
Run: python verify_tables.py
"""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results" / "metrics_aggregated.json"

HEADER = ["Seed", "Episodes", "Baseline", "Eval return", "Length",
          "Terminal", "Tail mean", "Overload", "Recall", "A / N / R"]
WIDTHS = [5, 9, 9, 12, 9, 9, 10, 9, 8, 22]


def q(value: float, places: int) -> Decimal:
    return Decimal(str(value)).quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)


def _row(cells: list[str]) -> str:
    return "  ".join(str(c).rjust(w) for c, w in zip(cells, WIDTHS))


def main() -> None:
    data = json.loads(RESULTS.read_text())

    print(f"Paper Table 3, read from {RESULTS}")
    print(f"Seeds {data['seeds']}, {data['total_env_steps']:,} environment steps per seed.\n")
    print(_row(HEADER))
    line = "-" * (sum(WIDTHS) + 2 * (len(WIDTHS) - 1))
    print(line)

    for m in data["per_seed"]:
        c = m["trained_action_counts"]
        print(_row([
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
        ]))

    print(line)

    b, e, ln = data["baseline_pure_awake_return"], data["trained_eval_return"], data["trained_eval_length"]
    tm, ov, rc = data["tail_mean_return"], data["mean_overload"], data["mean_recall"]
    a, n, r = data["action_AWAKE"], data["action_NREM"], data["action_REM"]

    print(_row([
        "Mean", "", f"{q(b['mean'], 2)}", f"{q(e['mean'], 2)}", f"{q(ln['mean'], 2)}",
        "", f"{q(tm['mean'], 2)}", f"{q(ov['mean'], 3)}", f"{q(rc['mean'], 3)}",
        f"{q(a['mean'], 1)} / {q(n['mean'], 1)} / {q(r['mean'], 1)}",
    ]))
    print(_row([
        "SD", "", f"+/-{q(b['sd'], 2)}", f"+/-{q(e['sd'], 2)}", f"+/-{q(ln['sd'], 2)}",
        "", f"+/-{q(tm['sd'], 2)}", f"+/-{q(ov['sd'], 3)}", f"+/-{q(rc['sd'], 3)}",
        f"+/-{q(a['sd'], 1)} / +/-{q(n['sd'], 1)} / +/-{q(r['sd'], 1)}",
    ]))


if __name__ == "__main__":
    main()
