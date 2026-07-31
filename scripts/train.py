"""Deploy the production model (trailing-window) and run walk-forward verification."""
from eml.models.price_forecast import train, walk_forward

if __name__ == "__main__":
    d = train()
    print("=== deploy (trailing-window regime-tracking model) ===")
    print(f"trained on {d['deploy_rows']} rows  {d['deploy_span'][0][:10]} -> {d['deploy_span'][1][:10]}")
    print(f"bias correction {d['bias']:+} EUR  |  CQR offset +/-{d['cqr_offset']} EUR")

    print("\n=== walk-forward verification (retrain every block) ===")
    m = walk_forward()
    print(f"blocks={m['blocks']}  window={m['window_days']}d  step={m['step_days']}d  "
          f"[{m['test_span'][0][:10]} -> {m['test_span'][1][:10]}]")
    print(f"MAE {m['mae']}  RMSE {m['rmse']}  coverage {m['p10_p90_coverage_pct']}%  "
          f"bias {m['bias_eur']:+} EUR/MWh")
