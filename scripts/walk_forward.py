"""Run walk-forward verification and print the resulting scorecard.

Usage:  python scripts/walk_forward.py [step_days]
Retrains on a trailing window every `step_days` and forecasts the next block, so every point is
a genuine, regime-tracking out-of-sample forecast. Freezes results for the dashboard verification.
"""
import sys

from eml.models.price_forecast import walk_forward
from eml.models.verify import scorecard

if __name__ == "__main__":
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    m = walk_forward(step_days=step)
    print(f"walk-forward: {m['blocks']} blocks, window {m['window_days']}d, step {step}d")
    s = scorecard()
    print(f"=== {s['zone']} DAM  [{s['window'][0][:10]} -> {s['window'][1][:10]}]  {s['n_days']} days ===")
    print(f"MAE {s['mae']}  RMSE {s['rmse']}  bias {s['bias']:+} EUR/MWh")
    print(f"P10-P90 coverage {s['coverage_pct']}%  |  skill vs persistence {s['skill_pct']:+}%  "
          f"(persistence MAE {s['persistence_mae']})")
