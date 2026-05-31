"""Migration 001 — initial schema.

This is a no-op migration that establishes schema_version = 1.
All registry files created by registry.register() already carry
schema_version = 1, so this migration only fires on files that were
written by hand or by a pre-versioning tool.

To add a new field to ModelConfig:
  1. Add the field with a default to src/mole_jepa/config.py.
  2. Bump CURRENT_SCHEMA_VERSION in src/mole_jepa/registry.py.
  3. Write apps/registry/migrations/m00N_<description>.py with:

         DESCRIPTION = "Add <field> (default <value>)"
         PRODUCES_VERSION = N

         def up(data: dict) -> dict:
             for entry in data["models"].values():
                 entry["config"].setdefault("<field>", <default>)
             data["schema_version"] = PRODUCES_VERSION
             return data

  4. Run: uv run python apps/registry/migrate.py --registry-path $REGISTRY_PATH
"""

DESCRIPTION = "Establish initial schema (no-op)"
PRODUCES_VERSION = 1


def up(data: dict) -> dict:
    """Apply this migration to *data* and return the updated dict."""
    data["schema_version"] = PRODUCES_VERSION
    return data
