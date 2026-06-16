"""Register all sweep configs and print (or submit) sbatch commands.

Usage::

    # Register configs and print commands
    uv run python scripts/run_sweep.py

    # Register and submit immediately
    uv run python scripts/run_sweep.py --submit
"""

import argparse
import subprocess
import sys

sys.path.insert(0, "src")

from mole_jepa import registry
from mole_jepa.config import ModelConfig

# ── shared base ───────────────────────────────────────────────────────────────

_BASE = dict(
    embed_dim=256,
    predictor_hidden_dim=256,
    predictor_n_layers=2,
    image_encoder_model_name="google/vit-base-patch16-224",
    text_encoder_model_name="answerdotai/ModernBERT-base",
    contrastive=False,
    reverse_predictor=False,
    jepa_regularize_z_i=True,
    jepa_regularize_z_t=True,
    attn_implementation="sdpa",
    freeze_image_encoder=False,
)

_LAMBDAS = [0.25, 0.5, 1.0, 2.0]


def _build_configs(tag: str) -> list[tuple[str, dict]]:
    """Build the sweep config list, with *tag* appended to every name.

    Args:
        tag: Suffix appended to each config name (e.g. a version or date
            string) so reruns get fresh registry entries instead of
            colliding with a previous sweep's (possibly stale) checkpoints.

    Returns:
        List of ``(name, kwargs)`` tuples.
    """
    configs: list[tuple[str, dict]] = []
    for lam in _LAMBDAS:
        lam_tag = str(lam).replace(".", "")

        # Spherical: cosine loss + Wang-Isola uniformity
        configs.append(
            (
                f"sweep_sphere_lam{lam_tag}_{tag}",
                {
                    **_BASE,
                    "jepa_cosine_loss": True,
                    "jepa_spherical_uniformity": True,
                    "jepa_lam_image": lam,
                    "jepa_lam_text": lam,
                },
            )
        )

        # Gaussian: MSE + SIGReg (1024 directions)
        configs.append(
            (
                f"sweep_gauss_lam{lam_tag}_{tag}",
                {
                    **_BASE,
                    "jepa_cosine_loss": False,
                    "jepa_spherical_uniformity": False,
                    "jepa_lam_image": lam,
                    "jepa_lam_text": lam,
                    "sigreg_n_directions": 1024,
                    "sigreg_dist": "gaussian",
                    "sigreg_demean": False,
                },
            )
        )
    return configs


def main() -> None:
    """Register configs and print/submit sbatch commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default="v1",
        metavar="TAG",
        help=(
            "Suffix appended to every config name (default: 'v1'). "
            "Bump this on reruns (e.g. --tag v2) so new registry entries "
            "and checkpoint dirs don't collide with a previous sweep."
        ),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs via sbatch immediately after registering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing registry entries.",
    )
    args = parser.parse_args()

    configs = _build_configs(args.tag)

    print(f"Registering {len(configs)} sweep configs (tag={args.tag!r}) …\n")
    for name, kwargs in configs:
        cfg = ModelConfig(**kwargs)
        geometry = "sphere" if kwargs.get("jepa_spherical_uniformity") else "gaussian"
        lam = kwargs["jepa_lam_image"]
        desc = f"Sweep: {geometry}, λ={lam}, embed_dim=256"
        try:
            registry.register(name, cfg, description=desc, overwrite=args.overwrite)
            print(f"  ✓ {name}")
        except ValueError:
            print(
                f"  ~ {name}  (already registered, skip — pass --overwrite to replace)"
            )

    print("\nsbatch commands:\n")
    commands = [f"sbatch scripts/sweep_train.sh --config {name}" for name, _ in configs]
    for cmd in commands:
        print(f"  {cmd}")

    if args.submit:
        print("\nSubmitting …")
        for cmd in commands:
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            status = result.stdout.strip() or result.stderr.strip()
            print(f"  {cmd.split('--config')[1].strip():30s}  {status}")


if __name__ == "__main__":
    main()
