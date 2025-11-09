"""Generate synthetic hierarchical data and export a lightweight Markdown report."""

from collections.abc import Iterable, Sequence
import numpy as np
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from plotly.graph_objs import Figure

from bayes_tools.helpers.synthetic_data_helpers import make_hierarchical_ou_dataset
from bayes_tools.helpers.visualisation_helpers import (
    plot_headcount_vs_productivity,
    plot_metric_timeseries,
    plot_productivity_vs_survey,
    plot_survey_heatmap,
)


MAX_IMAGE_BYTES = 300_000
"""Upper bound for exported chart assets to avoid bloated commits."""


@dataclass(frozen=True)
class ChartSpec:
    """Description of a chart and its on-disk representation."""

    filename: str
    title: str
    figure: Figure


def _select_groups(codes: Sequence[str], limit: int) -> list[str]:
    """Return the first ``limit`` unique codes while preserving order."""

    seen: set[str] = set()
    selected: list[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            selected.append(code)
        if len(selected) >= limit:
            break
    return selected


def _export_figures(specs: Iterable[ChartSpec], output_dir: Path) -> list[Path]:
    """Serialise Plotly figures to compact PNG files.

    Using PNG keeps the byte-size low while remaining widely compatible.
    The generated images are returned in the order they were provided.
    """

    output_paths: list[Path] = []
    for spec in specs:
        path = output_dir / spec.filename
        _render_plotly_with_matplotlib(spec.figure, path)
        _constrain_png_size(path)
        output_paths.append(path)
    return output_paths


def _figures_to_markdown(
    charts: Sequence[ChartSpec],
    image_paths: Sequence[Path],
    markdown_path: Path,
    *,
    seed: int,
    rows: int,
) -> None:
    """Write a Markdown report embedding the generated figures."""

    heading = "# Synthetic Hierarchical Report"
    metadata = [
        heading,
        "",
        f"- Seed: `{seed}`",
        f"- Rows: `{rows:,}`",
        "",
        "Generated visuals:",
        "",
    ]

    figure_sections: list[str] = []
    for chart, image_path in zip(charts, image_paths, strict=True):
        size_kb = image_path.stat().st_size / 1024
        figure_sections.extend(
            [
                f"## {chart.title}",
                "",
                f"![{chart.title}]({image_path.name})",
                f"> File size: {size_kb:.1f} KiB",
                "",
            ]
        )

    markdown_path.write_text("\n".join(metadata + figure_sections), encoding="utf-8")


def _render_plotly_with_matplotlib(fig: Figure, path: Path) -> None:
    """Render a Plotly figure as a static PNG using Matplotlib primitives."""

    layout = fig.layout
    plt_fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

    title = getattr(layout.title, "text", "") if layout.title else ""
    if title:
        ax.set_title(title)

    x_title = getattr(layout.xaxis, "title", None)
    if x_title and getattr(x_title, "text", ""):
        ax.set_xlabel(x_title.text)

    y_title = getattr(layout.yaxis, "title", None)
    if y_title and getattr(y_title, "text", ""):
        ax.set_ylabel(y_title.text)

    has_legend = False

    for trace in fig.data:
        if getattr(trace, "type", "") == "heatmap":
            _draw_heatmap(ax, trace)
        elif getattr(trace, "type", "") == "scatter":
            handle, label = _draw_scatter(ax, trace)
            if handle is not None and label:
                has_legend = True

    if has_legend:
        ax.legend(loc="best")

    plt_fig.tight_layout()
    plt_fig.savefig(
        path,
        bbox_inches="tight",
        dpi=110,
        pil_kwargs={"optimize": True},
    )
    plt.close(plt_fig)


def _constrain_png_size(path: Path, *, max_bytes: int = MAX_IMAGE_BYTES) -> None:
    """Resample an image until its size stays below ``max_bytes``."""

    try:
        current_size = path.stat().st_size
    except FileNotFoundError:
        return
    if current_size <= max_bytes:
        return

    with Image.open(path) as image:
        image = _ensure_rgba(image)
        image.save(path, optimize=True)
        for _ in range(8):
            if path.stat().st_size <= max_bytes:
                break
            new_width = max(1, int(image.width * 0.85))
            new_height = max(1, int(image.height * 0.85))
            if (new_width, new_height) == image.size:
                break
            image = image.resize((new_width, new_height), Image.LANCZOS)
            image.save(path, optimize=True)


def _ensure_rgba(image: Image.Image) -> Image.Image:
    """Return an RGBA copy of ``image`` to preserve transparency information."""

    if image.mode in {"RGB", "RGBA"}:
        return image
    return image.convert("RGBA")


def _draw_heatmap(ax: plt.Axes, trace: object) -> None:
    """Draw a heatmap trace onto the provided axes."""

    z = np.array(getattr(trace, "z", []), dtype=float)
    colorscale = getattr(trace, "colorscale", None)
    cmap = _resolve_cmap(colorscale)
    im = ax.imshow(z, aspect="auto", cmap=cmap, origin="lower")

    x_vals_raw = getattr(trace, "x", [])
    y_vals_raw = getattr(trace, "y", [])
    x_vals = x_vals_raw.tolist() if isinstance(x_vals_raw, np.ndarray) else list(x_vals_raw)
    y_vals = y_vals_raw.tolist() if isinstance(y_vals_raw, np.ndarray) else list(y_vals_raw)
    if x_vals:
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([str(v) for v in x_vals], rotation=45, ha="right")
    if y_vals:
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([str(v) for v in y_vals])

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _draw_scatter(ax: plt.Axes, trace: object) -> tuple[object | None, str]:
    """Draw scatter/line traces and return the Matplotlib handle and label."""

    mode = getattr(trace, "mode", "lines")
    name = getattr(trace, "name", "")
    x = getattr(trace, "x", [])
    y = getattr(trace, "y", [])
    x_values = pd.to_datetime(x) if _looks_like_dates(x) else x

    marker = getattr(trace, "marker", None)
    line = getattr(trace, "line", None)
    color = getattr(line, "color", None) if line else None
    if color is None and marker is not None:
        color = getattr(marker, "color", None)
    if isinstance(color, (list, tuple, np.ndarray)):
        color = None

    linestyle = "-"
    if line and getattr(line, "dash", "") == "dash":
        linestyle = "--"

    label_used = False
    handle: object | None = None

    if "lines" in mode:
        (handle,) = ax.plot(
            x_values,
            y,
            label=name or None,
            color=color,
            linestyle=linestyle,
        )
        label_used = bool(name)

    if "markers" in mode:
        marker_size = getattr(marker, "size", None) if marker is not None else None
        marker_symbol = getattr(marker, "symbol", None) if marker is not None else None
        marker_color = getattr(marker, "color", color) if marker is not None else color

        sizes = _normalise_sizes(marker_size)
        array_color = marker_color if isinstance(marker_color, (list, tuple, np.ndarray)) else None
        cmap = _resolve_cmap(getattr(marker, "colorscale", None)) if (marker and array_color is not None) else None

        scatter = ax.scatter(
            x_values,
            y,
            s=sizes,
            label=None if label_used else name or None,
            color=None if array_color is not None else marker_color,
            c=array_color,
            cmap=cmap,
            marker=_resolve_marker(marker_symbol),
            alpha=0.85,
            edgecolors="black" if marker_symbol == "circle" else None,
        )
        if array_color is not None and cmap is not None:
            plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        handle = handle or scatter
        label_used = label_used or bool(name)

    return handle, name if label_used else ""


def _normalise_sizes(size: object) -> np.ndarray | float | None:
    """Convert Plotly marker sizes to sensible Matplotlib equivalents."""

    if size is None:
        return None
    if isinstance(size, (int, float)):
        return float(size) * 5.0
    arr = np.array(size, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return None
    max_val = np.nanmax(arr)
    if max_val == 0:
        return np.clip(arr, 20, None)
    scaled = (arr / max_val) * 300
    return np.clip(scaled, 20, None)


def _resolve_marker(symbol: object) -> str:
    """Map Plotly marker symbols onto Matplotlib markers."""

    mapping = {
        "circle": "o",
        "square": "s",
        "diamond": "D",
        "triangle-up": "^",
    }
    if isinstance(symbol, str) and symbol in mapping:
        return mapping[symbol]
    return "o"


def _resolve_cmap(colorscale: object) -> str:
    """Convert a Plotly colorscale identifier into a Matplotlib colormap name."""

    if isinstance(colorscale, str):
        return colorscale.lower()
    return "viridis"


def _looks_like_dates(values: Sequence[object]) -> bool:
    """Return ``True`` if the sequence appears to contain datetime values."""

    if values is None:
        return False
    if isinstance(values, np.ndarray):
        if values.size == 0:
            return False
        sample = values.flat[0]
    else:
        if len(values) == 0:  # type: ignore[arg-type]
            return False
        sample = values[0]  # type: ignore[index]
    if isinstance(sample, (pd.Timestamp, np.datetime64)):
        return True
    if isinstance(sample, str):
        try:
            pd.to_datetime(sample)
            return True
        except (ValueError, TypeError):
            return False
    return False


def build_report(output_dir: Path, *, seed: int = 42) -> dict[str, object]:
    """Generate synthetic data, charts, and a Markdown report."""

    output_dir.mkdir(parents=True, exist_ok=True)

    df: pd.DataFrame = make_hierarchical_ou_dataset(seed=seed)
    data_path = output_dir / "hierarchical_panel.csv.gz"
    df.to_csv(data_path, index=False, compression="gzip")

    ou_groups = _select_groups(df["ou_code"], 4)
    site_groups = _select_groups(df["site_id"], 5)
    region_groups = _select_groups(df["region_id"], 3)

    charts = [
        ChartSpec(
            filename="metric_timeseries.png",
            title="Productivity trend",
            figure=plot_metric_timeseries(
                df,
                level="ou",
                groups=ou_groups,
                rolling_window=3,
            ),
        ),
        ChartSpec(
            filename="productivity_vs_survey.png",
            title="Survey vs productivity",
            figure=plot_productivity_vs_survey(
                df,
                level="site",
                groups=site_groups,
            ),
        ),
        ChartSpec(
            filename="survey_heatmap.png",
            title="Survey heatmap",
            figure=plot_survey_heatmap(
                df,
                level="site",
                groups=site_groups,
                value="survey_score",
            ),
        ),
        ChartSpec(
            filename="headcount_vs_productivity.png",
            title="Productivity vs FTE",
            figure=plot_headcount_vs_productivity(
                df,
                level="region",
                groups=region_groups,
            ),
        ),
    ]

    image_paths = _export_figures(charts, output_dir)
    markdown_path = output_dir / "synthetic_report.md"
    _figures_to_markdown(
        charts,
        image_paths,
        markdown_path,
        seed=seed,
        rows=len(df),
    )

    return {
        "data": data_path,
        "report": markdown_path,
        "images": tuple(image_paths),
    }


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/output"),
        help="Directory where artefacts will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the synthetic data generator.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    outputs = build_report(args.output, seed=args.seed)
    print(f"Dataset saved to: {outputs['data']}")
    print(f"Markdown report saved to: {outputs['report']}")


if __name__ == "__main__":
    main()
