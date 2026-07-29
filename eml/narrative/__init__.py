"""Layer 6 (AI): turn forecasts into natural-language answers to Why? and What next?.

Uses the price model's SHAP contributions (LightGBM pred_contrib) to attribute each forecast
to its drivers, then renders a templated morning brief. Deterministic and free; an LLM can
later rewrite the same structured facts into richer prose (drop-in — same inputs).
"""
