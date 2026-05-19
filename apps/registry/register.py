r"""CLI for registering, listing, and deregistering models in the NFS registry.

Wraps :func:`mole_jepa.nfs_registry.register` so you can manage the registry
from the cluster shell without writing a Python script.

Usage examples::

    # List all registered models
    uv run python apps/registry/register.py --list

    # Register a new model by cloning config from an existing entry
    uv run python apps/registry/register.py \\
        --name vit_base_bert_jepa_frozen_v2 \\
        --from-entry vit_base_bert_jepa_frozen \\
        --description "Lower LR experiment" \\
        --checkpoint-base $CHECKPOINT_DIR

    # Register a model by supplying the full config as JSON
    uv run python apps/registry/register.py \\
        --name my_experiment \\
        --config-json '{"embed_dim": 128, "contrastive": false, ...}' \\
        --checkpoint-base $CHECKPOINT_DIR \\
        --description "Tiny model ablation"

    # Point at an existing checkpoint directory (e.g. after import)
    uv run python apps/registry/register.py \\
        --name vit_base_bert_jepa_frozen \\
        --from-entry vit_base_bert_jepa_frozen \\
        --checkpoint-dir /scratch/vpathak/checkpoints/abc123def456

    # Remove an entry (does NOT delete checkpoint files)
    uv run python apps/registry/register.py --deregister vit_base_bert_jepa_frozen

All commands respect --registry-path (or $REGISTRY_PATH).
"""

import argparse
import json
import os
import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from mole_jepa import config as config_module  # noqa: E402
from mole_jepa import nfs_registry  # noqa: E402


def _cmd_list(registry_dir: str) -> None:
    """Print all registered models."""
    entries = nfs_registry.list_entries(registry_dir)
    if not entries:
        print("Registry is empty.")
        return
    for entry in entries:
        print(f"{entry.name}")
        print(f"  checkpoint_dir : {entry.checkpoint_dir}")
        print(f"  created_at     : {entry.created_at}")
        if entry.description:
            print(f"  description    : {entry.description}")
        cfg = entry.config
        print(
            f"  config         : embed_dim={cfg.embed_dim}  "
            f"contrastive={cfg.contrastive}  "
            f"freeze_image_encoder={cfg.freeze_image_encoder}"
        )
        print()


def _cmd_register(args: argparse.Namespace) -> None:
    """Register a new model entry."""
    # ── resolve config ────────────────────────────────────────────────────────
    if args.from_entry and args.config_json:
        raise SystemExit("--from-entry and --config-json are mutually exclusive.")
    if not args.from_entry and not args.config_json:
        raise SystemExit("Provide either --from-entry or --config-json.")

    if args.from_entry:
        source = nfs_registry.get_entry(args.from_entry, args.registry_path)
        model_config = source.config
    else:
        try:
            fields = json.loads(args.config_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --config-json: {exc}") from exc
        try:
            model_config = config_module.ModelConfig(**fields)
        except TypeError as exc:
            raise SystemExit(f"Invalid ModelConfig fields: {exc}") from exc

    # ── resolve checkpoint path ───────────────────────────────────────────────
    if args.checkpoint_dir and args.checkpoint_base:
        raise SystemExit(
            "--checkpoint-dir and --checkpoint-base are mutually exclusive."
        )
    if not args.checkpoint_dir and not args.checkpoint_base:
        raise SystemExit("Provide either --checkpoint-base or --checkpoint-dir.")

    entry = nfs_registry.register(
        args.name,
        model_config,
        checkpoint_base=args.checkpoint_base,
        checkpoint_dir=args.checkpoint_dir,
        description=args.description or "",
        registry_dir=args.registry_path,
        overwrite=args.overwrite,
    )
    print(f"Registered {entry.name!r}")
    print(f"  checkpoint_dir : {entry.checkpoint_dir}")
    print(f"  config hash    : {entry.config.serialize()}")


def _cmd_deregister(name: str, registry_dir: str) -> None:
    """Remove an entry from the registry."""
    nfs_registry.deregister(name, registry_dir)
    print(f"Removed {name!r} from registry (checkpoint files untouched).")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Manage the MoLeJEPA NFS model registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--registry-path",
        default=os.environ.get("REGISTRY_PATH"),
        metavar="DIR",
        help="Registry directory. Defaults to $REGISTRY_PATH.",
    )

    # ── mutually exclusive top-level commands ─────────────────────────────────
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--list",
        action="store_true",
        help="List all registered models.",
    )
    mode.add_argument(
        "--name",
        metavar="NAME",
        help="Name of the model to register.",
    )
    mode.add_argument(
        "--deregister",
        metavar="NAME",
        help="Remove a model from the registry (does not delete checkpoint files).",
    )

    # ── registration options (used with --name) ───────────────────────────────
    parser.add_argument(
        "--from-entry",
        metavar="NAME",
        help="Clone the config from an existing registry entry.",
    )
    parser.add_argument(
        "--config-json",
        metavar="JSON",
        help="Full ModelConfig as a JSON object string.",
    )
    parser.add_argument(
        "--checkpoint-base",
        default=os.environ.get("CHECKPOINT_DIR"),
        metavar="DIR",
        help=(
            "Root checkpoint directory; hash subdir is appended automatically. "
            "Defaults to $CHECKPOINT_DIR."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        metavar="DIR",
        help="Explicit checkpoint directory (overrides --checkpoint-base).",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Optional human-readable note stored with the entry.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing entry with the same name.",
    )

    args = parser.parse_args()

    if not args.registry_path:
        raise SystemExit(
            "No registry path set. "
            "Pass --registry-path or set $REGISTRY_PATH in your environment."
        )

    if args.list:
        _cmd_list(args.registry_path)
    elif args.deregister:
        _cmd_deregister(args.deregister, args.registry_path)
    else:
        _cmd_register(args)


if __name__ == "__main__":
    main()
