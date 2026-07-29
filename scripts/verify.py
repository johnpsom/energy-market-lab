"""Print the forecast-verification scorecard (frozen forecasts vs realized prices)."""
from eml.models.verify import scorecard

if __name__ == "__main__":
    s = scorecard()
    if not s.get("available"):
        raise SystemExit("No frozen forecasts yet — run scripts/train.py (freezes the "
                         "out-of-sample test period) or scripts/forecast.py first.")
    print(f"=== Verification — {s['zone']} DAM  [{s['window'][0][:10]} -> {s['window'][1][:10]}] ===")
    print(f"days={s['n_days']}  hours={s['n_hours']}")
    print(f"MAE  {s['mae']}  RMSE {s['rmse']}  bias {s['bias']:+} EUR/MWh")
    print(f"P10-P90 coverage {s['coverage_pct']}%  (target ~80%)")
    print(f"pinball {s['pinball']}")
    print(f"skill vs persistence: {s['skill_pct']:+}%  (persistence MAE {s['persistence_mae']})")
    if s["spike_brier"] is not None:
        print(f"spike-probability Brier score: {s['spike_brier']}  (lower is better)")
    print("\nlast 5 days:")
    for d in s["daily"][-5:]:
        print(f"  {d['date']}  MAE {d['mae']:5}  cov {d['cov']:5}%  "
              f"fcst {d['fcst_avg']:6} vs actual {d['actual_avg']:6}")
