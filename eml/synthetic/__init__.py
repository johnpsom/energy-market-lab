"""Synthetic-fundamentals bridge.

Populates the ENTSO-E-shaped warehouse tables (load, generation, prices) with a transparent
merit-order model driven by REAL Open-Meteo weather. Lets the full M3->M5 spine run before the
ENTSO-E token lands. Rows are tagged source='synthetic' so real data can replace them cleanly.
"""
