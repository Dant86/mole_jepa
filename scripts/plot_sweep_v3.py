"""Generate a static Plotly scatter: MSE vs uniformity coloured by i2t R@1.

Usage:
    uv run python scripts/plot_sweep_v3.py               # writes sweep_v3_scatter.html
    uv run python scripts/plot_sweep_v3.py --out my.html
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

_CSV = Path(__file__).parent.parent / "sweep_v3_history.csv"

# Map run names to readable labels
_LABELS = {
    "sweep_sphere_lam025_v3": "sphere λ=0.25",
    "sweep_sphere_lam05_v3": "sphere λ=0.5",
    "sweep_sphere_lam10_v3": "sphere λ=1.0",
    "sweep_sphere_lam20_v3": "sphere λ=2.0",
    "sweep_gauss_lam025_v3": "gauss λ=0.25",
    "sweep_gauss_lam05_v3": "gauss λ=0.5",
    "sweep_gauss_lam10_v3": "gauss λ=1.0",
    "sweep_gauss_lam20_v3": "gauss λ=2.0",
}

_MARKERS = {
    "sphere": "circle",
    "gauss": "diamond",
}

_PALETTE = {
    0.25: "#a8d8ea",
    0.5: "#4a90d9",
    1.0: "#2c5f8a",
    2.0: "#0d2b45",
}


def main() -> None:
    """Parse args and render the scatter plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(_CSV))
    parser.add_argument("--out", default="sweep_v3_scatter.html")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df.dropna(
        subset=["epoch/train_loss_mse", "epoch/train_loss_reg_text", "retrieval/i2t_r1"]
    )

    r1_min = df["retrieval/i2t_r1"].min()
    r1_max = df["retrieval/i2t_r1"].max()

    fig = go.Figure()

    for run, group in df.groupby("run"):
        group = group.sort_values("epoch")
        run_str = str(run)
        label = _LABELS.get(run_str, run_str)
        geometry = "sphere" if "sphere" in run_str else "gauss"
        lam = float(group["lam"].iloc[0])
        line_color = _PALETTE.get(lam, "#888888")

        fig.add_trace(
            go.Scatter(
                x=group["epoch/train_loss_mse"],
                y=group["epoch/train_loss_reg_text"],
                mode="lines+markers",
                name=label,
                text=[
                    f"<b>{label}</b><br>epoch {int(r['epoch'])}<br>"
                    f"MSE={r['epoch/train_loss_mse']:.4f}<br>"
                    f"Reg={r['epoch/train_loss_reg_text']:.4f}<br>"
                    f"i2t R@1={r['retrieval/i2t_r1']:.3f}"
                    for _, r in group.iterrows()
                ],
                hoverinfo="text",
                line=dict(
                    color=line_color,
                    width=1.5,
                    dash="dot" if geometry == "gauss" else "solid",
                ),
                marker=dict(
                    symbol=_MARKERS[geometry],
                    size=10,
                    color=group["retrieval/i2t_r1"].tolist(),
                    colorscale="RdYlGn",
                    cmin=r1_min,
                    cmax=r1_max,
                    showscale=True if run == list(df["run"].unique())[-1] else False,
                    colorbar=dict(
                        title="i2t R@1",
                        thickness=14,
                        len=0.6,
                    ),
                    line=dict(color=line_color, width=1),
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="MoLeJEPA v3 sweep — MSE vs uniformity (colour = i2t R@1)",
            font=dict(size=15),
        ),
        xaxis=dict(title="MSE loss", type="linear", tickformat=".4f"),
        yaxis=dict(title="Uniformity / reg loss"),
        legend=dict(
            title="Run",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=12),
        hovermode="closest",
        width=900,
        height=580,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)

    out = Path(args.out)
    pio.write_html(fig, str(out), include_plotlyjs=True, full_html=True)
    print(f"Saved → {out.resolve()}")


if __name__ == "__main__":
    main()
