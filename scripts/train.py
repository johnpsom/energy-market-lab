"""Train + backtest the DAM price forecaster; save artifacts and print the report."""
import json

from eml.models.price_forecast import train

if __name__ == "__main__":
    m = train()
    print("=== DAM price forecast - time-ordered holdout backtest ===")
    print(f"train: {m['train_span'][0]} -> {m['train_span'][1]}")
    print(f"test : {m['test_span'][0]} -> {m['test_span'][1]}  (n={m['n']})")
    print(f"MAE  : {m['mae']} EUR/MWh")
    print(f"RMSE : {m['rmse']} EUR/MWh")
    print(f"pinball loss: {m['pinball']}")
    print(f"P10-P90 coverage: {m['p10_p90_coverage_pct']}% (CQR-calibrated, target ~80%)"
          f"  |  before CQR: {m['p10_p90_coverage_uncalibrated_pct']}%  (offset +/-{m['cqr_offset_eur']} EUR)")
    print(f"spike>150 base rate: {m['spike_base_rate_pct']}%  | negative base rate: {m['neg_base_rate_pct']}%")
    print("\nfull metrics -> models/artifacts/metrics.json")
