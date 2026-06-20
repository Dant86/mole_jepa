"""Generate side-by-side Plotly scatters: MSE vs uniformity, coloured by i2t R@1.

Marker colour encodes i2t R@1 (RdYlGn).  Marker size encodes λ.
Left panel: spherical uniformity runs.  Right panel: Gaussian SIGReg runs.

Usage:
    uv run python scripts/plot_sweep_v3.py            # writes sweep_v3_scatter.png
    uv run python scripts/plot_sweep_v3.py --out my.png
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

_CSV = Path(__file__).parent.parent / "sweep_v3_history.csv"

_LABELS = {
    "sweep_sphere_lam025_v3": "λ=0.25",
    "sweep_sphere_lam05_v3": "λ=0.5",
    "sweep_sphere_lam10_v3": "λ=1.0",
    "sweep_sphere_lam20_v3": "λ=2.0",
    "sweep_gauss_lam025_v3": "λ=0.25",
    "sweep_gauss_lam05_v3": "λ=0.5",
    "sweep_gauss_lam10_v3": "λ=1.0",
    "sweep_gauss_lam20_v3": "λ=2.0",
}

# λ → marker diameter (px).  Perceptual area scaling: size ∝ sqrt(λ).
_LAM_SIZE = {0.25: 8, 0.5: 11, 1.0: 16, 2.0: 22}

_LINE_COLOR = "#555555"  # neutral — colour channel is reserved for R@1

_PANELS = {"sphere": 1, "gauss": 2}
_PANEL_TITLES = ("Spherical uniformity", "Gaussian SIGReg")


def main() -> None:
    """Parse args and render the side-by-side scatter plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(_CSV))
    parser.add_argument("--out", default="sweep_v3_scatter.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(
        subset=["epoch/train_loss_mse", "epoch/train_loss_reg_text", "retrieval/i2t_r1"]
    )

    r1_min = float(df["retrieval/i2t_r1"].min())
    r1_max = float(df["retrieval/i2t_r1"].max())

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=_PANEL_TITLES,
        shared_yaxes=False,
        horizontal_spacing=0.14,
    )

    # Which run is last in each panel — that one gets the R@1 colorbar.
    last_in_panel: dict[int, str] = {}
    for run, _ in df.groupby("run"):
        run_str = str(run)
        col = _PANELS["sphere" if "sphere" in run_str else "gauss"]
        last_in_panel[col] = run_str

    for run, group in df.groupby("run"):
        group = group.sort_values("epoch")
        run_str = str(run)
        label = _LABELS.get(run_str, run_str)
        geometry = "sphere" if "sphere" in run_str else "gauss"
        col = _PANELS[geometry]
        lam = float(group["lam"].iloc[0])
        marker_size = _LAM_SIZE.get(lam, 12)
        show_colorbar = run_str == last_in_panel[col]

        fig.add_trace(
            go.Scatter(
                x=group["epoch/train_loss_mse"],
                y=group["epoch/train_loss_reg_text"],
                mode="lines+markers",
                name=label,
                legendgroup=label,
                showlegend=False,  # legend replaced by size-legend dummy traces below
                text=[
                    f"<b>{label}</b><br>epoch {int(r['epoch'])}<br>"
                    f"MSE={r['epoch/train_loss_mse']:.4f}<br>"
                    f"Reg={r['epoch/train_loss_reg_text']:.4f}<br>"
                    f"i2t R@1={r['retrieval/i2t_r1']:.3f}"
                    for _, r in group.iterrows()
                ],
                hoverinfo="text",
                line=dict(color=_LINE_COLOR, width=1, dash="dot"),
                marker=dict(
                    symbol="circle",
                    size=marker_size,
                    color=group["retrieval/i2t_r1"].tolist(),
                    colorscale="RdYlGn",
                    cmin=r1_min,
                    cmax=r1_max,
                    showscale=show_colorbar,
                    colorbar=dict(
                        title=dict(text="i2t R@1", side="right"),
                        thickness=14,
                        len=0.75,
                        x=1.02,
                        y=0.5,
                    ),
                    line=dict(color="white", width=0.5),
                ),
            ),
            row=1,
            col=col,
        )

    # ── size legend: dummy invisible traces, one per λ value ─────────────────
    for lam, size in sorted(_LAM_SIZE.items()):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"λ = {lam}",
                legendgroup=f"lam_{lam}",
                showlegend=True,
                marker=dict(
                    symbol="circle",
                    size=size,
                    color="#888888",
                    line=dict(color="white", width=0.5),
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                "MoLeJEPA v3 sweep — MSE vs uniformity"
                "<br><sup>colour = i2t R@1 · size = λ</sup>"
            ),
            font=dict(size=14),
            x=0.5,
        ),
        legend=dict(
            title=dict(text="λ"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc",
            borderwidth=1,
            x=0.38,
            y=0.99,
            xanchor="right",
            yanchor="top",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        hovermode="closest",
        width=1100,
        height=500,
    )
    fig.update_xaxes(
        title_text="MSE loss",
        showgrid=True,
        gridcolor="#eeeeee",
        zeroline=False,
        tickformat=".4f",
    )
    fig.update_yaxes(
        title_text="Uniformity / reg loss",
        showgrid=True,
        gridcolor="#eeeeee",
        zeroline=False,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#eeeeee",
        zeroline=False,
        col=2,
    )

    out = Path(args.out)
    if out.suffix == ".html":
        pio.write_html(fig, str(out), include_plotlyjs=True, full_html=True)
    else:
        pio.write_image(fig, str(out), scale=2)
    print(f"Saved → {out.resolve()}")


if __name__ == "__main__":
    main()
