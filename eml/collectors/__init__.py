"""Layer 1-2: data acquisition. One module per source, all exposing `fetch(...)`.

The shared contract lets paid feeds (Montel, ECMWF, Bloomberg) slot in later behind the
same interface the free feeds use today.
"""
