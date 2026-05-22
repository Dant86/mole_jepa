r"""One-time import: migrate CONFIGS from registry.py into the NFS registry.

For each named config in ``src/mole_jepa/registry.py``:

1. Compute the SHA-256 hash (``config.serialize()``) — this is the name of
   the existing checkpoint subdirectory under ``--checkpoint-dir``.
2. Check whether that directory exists on disk.
3. Create an NFS registry entry pointing at the existing directory (no data
   is moved or renamed).

After a successful import and verification, delete ``src/mole_jepa/registry.py``
and remove the ``registry`` import from ``src/mole_jepa/__init__.py``.

Usage::

    # Dry run — print what would be imported, write nothing:
    uv run python apps/registry/import_from_registry_py.py \\
        --checkpoint-dir $CHECKPOINT_DIR \\
        --registry-path  $REGISTRY_PATH \\
        --dry-run

    # Live import:
    uv run python apps/registry/import_from_registry_py.py \\
        --checkpoint-dir $CHECKPOINT_DIR \\
        --registry-path  $REGISTRY_PATH

Entries that already exist in the NFS registry are skipped unless
--overwrite is passed.  Configs whose checkpoint directory does not exist
on disk are imported anyway (the checkpoint dir will be created when
training starts) unless --skip-missing is passed.
"""

import argparse
import os
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from mole_jepa import nfs_registry  # noqa: E402
from mole_jepa import registry as old_registry  # noqa: E402


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Import configs from registry.py into the NFS registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.environ.get("CHECKPOINT_DIR"),
        metavar="DIR",
        help=(
            "Root directory containing existing checkpoint subdirectories "
            "(named by config hash). Defaults to $CHECKPOINT_DIR."
        ),
    )
    parser.add_argument(
        "--registry-path",
        default=os.environ.get("REGISTRY_PATH"),
        metavar="DIR",
        help="NFS registry directory. Defaults to $REGISTRY_PATH.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be imported without writing anything.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing registry entries with the same name.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip configs whose checkpoint directory does not exist on disk.",
    )
    args = parser.parse_args()

    if not args.registry_path:
        raise SystemExit(
            "No registry path set. "
            "Pass --registry-path or set $REGISTRY_PATH in your environment."
        )
    if not args.checkpoint_dir:
        raise SystemExit(
            "No checkpoint directory set. "
            "Pass --checkpoint-dir or set $CHECKPOINT_DIR in your environment."
        )

    checkpoint_base = pathlib.Path(args.checkpoint_dir)
    configs = old_registry.CONFIGS

    print(f"Found {len(configs)} configs in registry.py")
    print(f"Checkpoint dir  : {checkpoint_base}")
    print(f"Registry path   : {args.registry_path}")
    print(f"Dry run         : {args.dry_run}")
    print()

    imported = 0
    skipped = 0

    for name, config in sorted(configs.items()):
        config_hash = config.serialize()
        checkpoint_dir = checkpoint_base / config_hash
        exists = checkpoint_dir.exists()

        status = "EXISTS" if exists else "MISSING"
        print(f"  {name}")
        print(f"    hash           : {config_hash[:16]}…")
        print(f"    checkpoint_dir : {checkpoint_dir}  [{status}]")

        if args.skip_missing and not exists:
            print("    → SKIPPED (--skip-missing)")
            skipped += 1
            continue

        if args.dry_run:
            print("    → would import")
            imported += 1
            continue

        try:
            nfs_registry.register(
                name,
                config,
                checkpoint_dir=checkpoint_base,  # base dir; hash subdir appended
                description=f"Imported from registry.py (hash: {config_hash[:8]})",
                registry_dir=args.registry_path,
                overwrite=args.overwrite,
            )
            print("    → imported")
            imported += 1
        except ValueError as exc:
            print(f"    → SKIPPED: {exc}")
            skipped += 1

        print()

    print(
        f"\nDone. "
        f"{'Would import' if args.dry_run else 'Imported'}: {imported}  "
        f"Skipped: {skipped}"
    )
    if not args.dry_run and imported:
        print(
            "\nNext steps:\n"
            "  1. Verify:  uv run python apps/registry/register.py --list\n"
            "  2. Delete:  src/mole_jepa/registry.py\n"
            "  3. Remove the 'registry' export from src/mole_jepa/__init__.py\n"
            "  4. Commit the removal."
        )


if __name__ == "__main__":
    main()
