r"""CLI for registering, listing, and deregistering models in the NFS registry.

Wraps :func:`mole_jepa.registry.register` so you can manage the registry
from the cluster shell without writing a Python script.

Usage examples::

    # List all registered models
    uv run python apps/registry/register.py --list

    # Register a model by cloning config from an existing entry
    uv run python apps/registry/register.py \
        --name vit_base_bert_jepa_frozen_v2 \
        --from-entry vit_base_bert_jepa_frozen \
        --description "Lower LR experiment"

    # Clone an entry and override specific fields
    uv run python apps/registry/register.py \
        --name vit_base_bert_jepa_laplace \
        --from-entry vit_base_bert_jepa_frozen \
        --override-json '{"sigreg_dist": "laplace"}' \
        --description "Laplace SIGReg ablation"

    # Register a model by supplying the full config as JSON
    uv run python apps/registry/register.py \
        --name my_experiment \
        --config-json '{"embed_dim": 128, "contrastive": false, ...}'  \
        --description "Tiny model ablation"

    # Override the checkpoint root (defaults to $CHECKPOINT_DIR)
    uv run python apps/registry/register.py \
        --name my_experiment \
        --config-json '{"embed_dim": 128}' \
        --checkpoint-dir /scratch/alt_checkpoints

    # Remove an entry (does NOT delete checkpoint files)
    uv run python apps/registry/register.py --deregister vit_base_bert_jepa_frozen

All commands default --registry-path to $REGISTRY_PATH and --checkpoint-dir
to $CHECKPOINT_DIR from the environment (or a .env file in the project root).
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
from mole_jepa import registry  # noqa: E402


def _cmd_list(registry_dir: str) -> None:
    """Print all registered models."""
    entries = registry.list_entries(registry_dir)
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
        source = registry.get_entry(args.from_entry, args.registry_path)
        model_config = source.config
        if args.override_json:
            try:
                overrides = json.loads(args.override_json)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid --override-json: {exc}") from exc
            try:
                import dataclasses

                model_config = dataclasses.replace(model_config, **overrides)
            except TypeError as exc:
                raise SystemExit(f"Invalid override fields: {exc}") from exc
    else:
        try:
            fields = json.loads(args.config_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --config-json: {exc}") from exc
        try:
            model_config = config_module.ModelConfig(**fields)
        except TypeError as exc:
            raise SystemExit(f"Invalid ModelConfig fields: {exc}") from exc

    entry = registry.register(
        args.name,
        model_config,
        checkpoint_dir=args.checkpoint_dir,  # None → falls back to $CHECKPOINT_DIR
        description=args.description or "",
        registry_dir=args.registry_path,  # None → falls back to $REGISTRY_PATH
        overwrite=args.overwrite,
    )
    print(f"Registered {entry.name!r}")
    print(f"  checkpoint_dir : {entry.checkpoint_dir}")
    print(f"  config hash    : {entry.config.serialize()}")


def _cmd_deregister(name: str, registry_dir: str | None) -> None:
    """Remove an entry from the registry."""
    registry.deregister(name, registry_dir)
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
        "--override-json",
        metavar="JSON",
        help=(
            "JSON object of ModelConfig fields to override when using --from-entry. "
            'E.g. \'{"sigreg_dist": "laplace"}\''
        ),
    )
    parser.add_argument(
        "--config-json",
        metavar="JSON",
        help="Full ModelConfig as a JSON object string.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        metavar="DIR",
        help=(
            "Root directory under which all model checkpoints live "
            "(e.g. /scratch/vpathak/checkpoints).  The config hash is "
            "appended automatically — you never need to specify the full path. "
            "Defaults to $CHECKPOINT_DIR."
        ),
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

    if args.list:
        _cmd_list(args.registry_path)
    elif args.deregister:
        _cmd_deregister(args.deregister, args.registry_path)
    else:
        _cmd_register(args)


if __name__ == "__main__":
    main()
