"""Create the warehouse schema. Idempotent."""
from eml.db import init_schema, settings

if __name__ == "__main__":
    init_schema()
    print(f"Schema created at: {settings.database_url}")
