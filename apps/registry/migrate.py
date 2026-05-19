"""Run pending schema migrations on the NFS registry.

Discovers migration modules in the ``migrations/`` subdirectory by the
pattern ``m[0-9][0-9][0-9]_*.py``, sorts them by their three-digit prefix,
and applies every migration whose ``PRODUCES_VERSION`` is greater than the
registry's current ``schema_version``.

Usage::

    uv run python apps/registry/migrate.py --registry-path $REGISTRY_PATH

    # Dry run (shows which migrations would run, writes nothing):
    uv run python apps/registry/migrate.py --registry-path $REGISTRY_PATH --dry-run

Writing a new migration
-----------------------
Create ``apps/registry/migrations/m00N_<description>.py`` with::

    DESCRIPTION = "Human-readable summary"
    PRODUCES_VERSION = N          # must equal the previous version + 1

    def up(data: dict) -> dict:
        # mutate data["models"] entries as needed, e.g.:
        for entry in data["models"].values():
            entry["config"].setdefault("new_field", default_value)
        data["schema_version"] = PRODUCES_VERSION
        return data

Then bump CURRENT_SCHEMA_VERSION in src/mole_jepa/nfs_registry.py to N
and run this script on the cluster.
"""

import argparse
import importlib.util
import os
import pathlib
import sys
import types

# Ensure the project root is on the path when run as a script.
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from mole_jepa.nfs_registry import _load, _Lock, _save  # noqa: E402


def _discover_migrations(
    migrations_dir: pathlib.Path,
) -> list[tuple[int, types.ModuleType]]:
    """Load and return all migration modules sorted by version number.

    Args:
        migrations_dir: Directory containing ``m00N_*.py`` files.

    Returns:
        List of ``(produces_version, module)`` pairs in ascending order.
    """
    results: list[tuple[int, types.ModuleType]] = []
    for path in sorted(migrations_dir.glob("m[0-9][0-9][0-9]_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        results.append((int(mod.PRODUCES_VERSION), mod))
    return results


def run_pending(
    registry_dir: str | pathlib.Path,
    *,
    dry_run: bool = False,
) -> int:
    """Apply all pending migrations to the registry.

    Args:
        registry_dir: Path to the registry directory on NFS.
        dry_run: If ``True``, print what would be done but do not write.

    Returns:
        Number of migrations applied (0 if already up to date).
    """
    migrations_dir = pathlib.Path(__file__).parent / "migrations"
    all_migrations = _discover_migrations(migrations_dir)

    with _Lock(registry_dir):
        data = _load(registry_dir)
        current = int(data.get("schema_version", 0))

        pending = [(v, m) for v, m in all_migrations if v > current]
        if not pending:
            print(f"Registry is up to date (schema_version={current}).")
            return 0

        print(f"Current schema_version: {current}. Pending migrations: {len(pending)}")
        for version, mod in pending:
            print(
                f"  {'[dry-run] ' if dry_run else ''}Applying {mod.__name__!r}: "
                f"{mod.DESCRIPTION!r}  →  schema_version={version}"
            )
            if not dry_run:
                data = mod.up(data)

        if not dry_run:
            _save(data, registry_dir)
            print(f"Done. schema_version: {current} → {data['schema_version']}")
        else:
            print("Dry run complete — no changes written.")

    return len(pending)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Apply pending schema migrations to the NFS registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--registry-path",
        default=os.environ.get("REGISTRY_PATH"),
        metavar="DIR",
        help="Registry directory. Defaults to $REGISTRY_PATH.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which migrations would run without writing anything.",
    )
    args = parser.parse_args()

    if not args.registry_path:
        raise SystemExit(
            "No registry path set. "
            "Pass --registry-path or set $REGISTRY_PATH in your environment."
        )

    run_pending(args.registry_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
