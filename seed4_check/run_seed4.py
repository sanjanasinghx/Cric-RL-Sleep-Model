"""Train seed 4 with the paper settings, writing only inside seed4_check/.

Uses run_sweep.run_one_seed rather than run_sweep.main, since main overwrites
results/, models/ and figures/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT))

import run_experiment as R  # noqa: E402
import run_sweep as S  # noqa: E402
from environment import Config  # noqa: E402

SEED = 4

# Point all output paths at seed4_check/.
S.RESULTS = OUT / "results"
S.MODELS = OUT / "models"
S.FIGURES = OUT / "figures"
R.FIGURES = OUT / "figures"
R.ARTIFACTS = OUT / "artifacts"

for d in (S.RESULTS, S.MODELS, S.FIGURES):
    d.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cfg = Config()

    print(f"seed {SEED}, {cfg.total_env_steps:,} env steps, horizon {cfg.max_steps}", flush=True)
    print(f"writing only under {OUT}", flush=True)

    result = S.run_one_seed(SEED, cfg, S.MODELS)
    metrics = result["metrics"]

    (S.RESULTS / "metrics_seed4.json").write_text(json.dumps(
        {"seeds": [SEED], "total_env_steps": cfg.total_env_steps, "per_seed": [metrics]},
        indent=2))
    result["log"].to_csv(S.RESULTS / "seed4_training_log.csv", index=False)

    print(f"\nwrote {S.RESULTS / 'metrics_seed4.json'}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
