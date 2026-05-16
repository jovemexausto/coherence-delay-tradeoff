from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq  # pyright: ignore[reportUnknownVariableType]
from scipy.stats import norm

from .common import export_rows_csv

BG = "#FDFEFE"
BLUE = "#1B4F72"
TEAL = "#148F77"
ORANGE = "#D35400"
PURPLE = "#6C3483"
RED = "#922B21"
GREY = "#707B7C"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    }
)


@dataclass(slots=True)
class GlueTheoremResearchConfig:
    sigma: float = 1.0
    fixed_span: float = 0.5
    fixed_span_H: float = 1.0
    n_values: tuple[int, ...] = (25, 50, 100, 200, 400)
    fixed_span_replications: int = 80
    reference_size: int = 30_000
    kappa_delta_n: int = 60
    kappa_delta_values: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)
    growth_betas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    growth_base_span: float = 0.25
    bounded_support_replications: int = 64


@dataclass(slots=True)
class GlueTheoremResearchResult:
    summary_rows: list[dict[str, str | float]]
    curve_rows: list[dict[str, str | float]]


def w2_sq_1d(
    sample_left: np.ndarray, sample_right: np.ndarray, n_q: int = 1024
) -> float:
    quantiles = (np.arange(n_q, dtype=float) + 0.5) / n_q
    left = np.quantile(sample_left, quantiles)
    right = np.quantile(sample_right, quantiles)
    return float(np.mean((left - right) ** 2))


def w2_1d(sample_left: np.ndarray, sample_right: np.ndarray, n_q: int = 1024) -> float:
    return float(np.sqrt(w2_sq_1d(sample_left, sample_right, n_q=n_q)))


def w2_sq_sample_to_mixture(
    sample: np.ndarray,
    means: np.ndarray,
    sigma: float,
    n_q: int = 512,
) -> float:
    quantiles = (np.arange(n_q, dtype=float) + 0.5) / n_q
    sample_q = np.quantile(sample, quantiles)
    mixture_q = mixture_quantiles(quantiles, means, sigma)
    return float(np.mean((sample_q - mixture_q) ** 2))


def w2_sq_sample_to_quantile_target(
    sample: np.ndarray,
    target_quantiles: np.ndarray,
    quantile_grid: np.ndarray,
) -> float:
    sample_q = np.quantile(sample, quantile_grid)
    return float(np.mean((sample_q - target_quantiles) ** 2))


def w2_sample_to_mixture(
    sample: np.ndarray,
    means: np.ndarray,
    sigma: float,
    n_q: int = 512,
) -> float:
    return float(np.sqrt(w2_sq_sample_to_mixture(sample, means, sigma, n_q=n_q)))


def w2_sample_to_quantile_target(
    sample: np.ndarray,
    target_quantiles: np.ndarray,
    quantile_grid: np.ndarray,
) -> float:
    return float(
        np.sqrt(
            w2_sq_sample_to_quantile_target(sample, target_quantiles, quantile_grid)
        )
    )


def fit_power_law_exponent(x_values: np.ndarray, y_values: np.ndarray) -> float:
    return float(-np.polyfit(np.log(x_values), np.log(y_values), 1)[0])


def fixed_span_means(n: int, span: float, H: float) -> np.ndarray:
    raw = np.arange(n, dtype=float) ** H
    raw -= raw.min()
    scale = raw.max() if raw.max() > 0.0 else 1.0
    return span * raw / scale


def mixture_cdf(x_value: float, means: np.ndarray, sigma: float) -> float:
    return float(np.mean(norm.cdf((x_value - means) / sigma)))


def mixture_pdf(x_value: float, means: np.ndarray, sigma: float) -> float:
    return float(np.mean(norm.pdf((x_value - means) / sigma) / sigma))


def mixture_quantiles(
    u_values: np.ndarray, means: np.ndarray, sigma: float
) -> np.ndarray:
    bracket_radius = float(np.max(np.abs(means)) + 8.0 * sigma + 1.0)
    lower = -bracket_radius
    upper = bracket_radius
    quantiles: list[float] = []
    for u in u_values:
        if u <= 0.0:
            quantiles.append(lower)
            continue
        if u >= 1.0:
            quantiles.append(upper)
            continue
        root = float(
            brentq(  # pyright: ignore[reportArgumentType]
                lambda x_value, target=u: mixture_cdf(x_value, means, sigma) - target,
                lower,
                upper,
                xtol=1e-10,
            )
        )
        quantiles.append(root)
    return np.asarray(quantiles, dtype=float)


def kappa_identity_value() -> float:
    """Numerical homogeneous-Gaussian baseline integral for the asymptotic constant."""
    grid = np.linspace(-8.0, 8.0, 20_001)
    integrand = norm.cdf(grid) * (1.0 - norm.cdf(grid)) / norm.pdf(grid)
    return float(np.trapezoid(integrand, grid))


def asymptotic_quantile_constants(
    means: np.ndarray,
    sigma: float,
    x_count: int = 1600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    radius = float(np.max(np.abs(means)) + 8.0 * sigma + 1.0)
    x_values = np.linspace(-radius, radius, x_count)
    f_values = np.asarray([mixture_pdf(x, means, sigma) for x in x_values], dtype=float)
    safe_f = np.maximum(f_values, 1e-10)
    fj_values = np.asarray(
        [norm.cdf((x_values - mean) / sigma) for mean in means],
        dtype=float,
    )
    fbar_cdf = np.mean(fj_values, axis=0)
    a_tri = np.mean(fj_values * (1.0 - fj_values), axis=0) / safe_f
    a_iid = fbar_cdf * (1.0 - fbar_cdf) / safe_f
    c2_tri = float(np.trapezoid(a_tri, x_values))
    c2_iid = float(np.trapezoid(a_iid, x_values))
    return x_values, a_tri, a_iid, c2_tri, c2_iid


def kappa_value(means: np.ndarray, sigma: float) -> float:
    _, _, _, c2_tri, _ = asymptotic_quantile_constants(means, sigma)
    return float(np.sqrt(c2_tri))


def verify_fixed_span_bound(
    config: GlueTheoremResearchConfig,
) -> tuple[
    list[dict[str, str | float]],
    list[dict[str, str | float]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    summary_rows: list[dict[str, str | float]] = []
    curve_rows: list[dict[str, str | float]] = []
    n_values = np.asarray(config.n_values, dtype=int)
    tri_errors: list[float] = []
    iid_errors: list[float] = []
    kappa_bounds: list[float] = []
    c2_values: list[float] = []

    for n in n_values:
        means = fixed_span_means(n, config.fixed_span, config.fixed_span_H)
        _, _, _, c2_tri, c2_iid = asymptotic_quantile_constants(means, config.sigma)
        kappa = float(np.sqrt(c2_tri))
        quantile_grid = (np.arange(512, dtype=float) + 0.5) / 512.0
        target_quantiles = mixture_quantiles(quantile_grid, means, config.sigma)

        tri_batch: list[float] = []
        iid_batch: list[float] = []
        for replication in range(config.fixed_span_replications):
            tri_rng = np.random.default_rng(20_000 + 100 * n + replication)
            iid_rng = np.random.default_rng(30_000 + 100 * n + replication)
            tri_sample = means + config.sigma * tri_rng.standard_normal(n)
            iid_index = iid_rng.integers(0, n, size=n)
            iid_sample = means[iid_index] + config.sigma * iid_rng.standard_normal(n)
            tri_batch.append(
                w2_sample_to_quantile_target(
                    tri_sample, target_quantiles, quantile_grid
                )
            )
            iid_batch.append(
                w2_sample_to_quantile_target(
                    iid_sample, target_quantiles, quantile_grid
                )
            )

        tri_mean = float(np.mean(tri_batch))
        iid_mean = float(np.mean(iid_batch))
        bound = kappa / np.sqrt(n)
        tri_errors.append(tri_mean)
        iid_errors.append(iid_mean)
        kappa_bounds.append(bound)
        c2_values.append(c2_tri)

        summary_rows.append(
            {
                "experiment": "fixed-span-bound",
                "n": n,
                "tri_mean_w2": round(tri_mean, 6),
                "iid_mean_w2": round(iid_mean, 6),
                "kappa_bound": round(bound, 6),
                "tri_over_kappa": round(tri_mean / bound, 6),
                "c2_tri": round(c2_tri, 6),
                "c2_iid": round(c2_iid, 6),
                "c2_ratio": round(c2_tri / c2_iid, 6),
            }
        )
        curve_rows.extend(
            [
                {
                    "experiment": "fixed-span-bound",
                    "setting": "triangular",
                    "n": n,
                    "value": round(tri_mean, 6),
                },
                {
                    "experiment": "fixed-span-bound",
                    "setting": "iid-mixture",
                    "n": n,
                    "value": round(iid_mean, 6),
                },
                {
                    "experiment": "fixed-span-bound",
                    "setting": "kappa-bound",
                    "n": n,
                    "value": round(bound, 6),
                },
            ]
        )

    tri_array = np.asarray(tri_errors, dtype=float)
    iid_array = np.asarray(iid_errors, dtype=float)
    bound_array = np.asarray(kappa_bounds, dtype=float)
    c2_array = np.asarray(c2_values, dtype=float)
    summary_rows.append(
        {
            "experiment": "fixed-span-rate",
            "tri_rate_a": round(fit_power_law_exponent(n_values, tri_array), 6),
            "iid_rate_a": round(fit_power_law_exponent(n_values, iid_array), 6),
            "max_tri_over_kappa": round(float(np.max(tri_array / bound_array)), 6),
            "span": config.fixed_span,
            "H": config.fixed_span_H,
        }
    )
    return summary_rows, curve_rows, n_values, tri_array, iid_array, c2_array


def kappa_vs_span_rows(
    config: GlueTheoremResearchConfig,
) -> tuple[list[dict[str, str | float]], np.ndarray, np.ndarray]:
    rows: list[dict[str, str | float]] = []
    span_values = np.asarray(config.kappa_delta_values, dtype=float)
    kappas: list[float] = []
    for span in span_values:
        means = fixed_span_means(config.kappa_delta_n, span, 1.0)
        kappa = kappa_value(means, config.sigma)
        kappas.append(kappa)
        rows.append(
            {
                "experiment": "kappa-vs-span",
                "span": round(span, 6),
                "kappa": round(kappa, 6),
            }
        )
    return rows, span_values, np.asarray(kappas, dtype=float)


def span_growth_rows(
    config: GlueTheoremResearchConfig,
) -> tuple[list[dict[str, str | float]], np.ndarray, np.ndarray]:
    rows: list[dict[str, str | float]] = []
    betas = np.asarray(config.growth_betas, dtype=float)
    exponents: list[float] = []
    n_values = np.asarray(config.n_values, dtype=int)
    for beta in betas:
        kappas: list[float] = []
        for n in n_values:
            span = config.growth_base_span * (n**beta)
            means = fixed_span_means(n, span, 1.0)
            kappas.append(kappa_value(means, config.sigma))
        kappa_array = np.asarray(kappas, dtype=float)
        kappa_exponent = float(
            np.polyfit(np.log(n_values.astype(float)), np.log(kappa_array), 1)[0]
        )
        error_exponent = 0.5 - kappa_exponent
        exponents.append(error_exponent)
        rows.append(
            {
                "experiment": "span-growth",
                "beta": round(beta, 6),
                "kappa_growth_exponent": round(kappa_exponent, 6),
                "predicted_error_exponent": round(error_exponent, 6),
            }
        )
    return rows, betas, np.asarray(exponents, dtype=float)


def bounded_support_inheritance_rows(
    config: GlueTheoremResearchConfig,
) -> tuple[list[dict[str, str | float]], list[dict[str, str | float]]]:
    rows: list[dict[str, str | float]] = []
    curve_rows: list[dict[str, str | float]] = []
    n_values = np.asarray(config.n_values, dtype=int)
    tri_errors: list[float] = []
    iid_errors: list[float] = []

    for n in n_values:
        means = fixed_span_means(n, config.fixed_span, config.fixed_span_H)
        ref_rng = np.random.default_rng(40_000 + n)
        ref_idx = ref_rng.integers(0, n, size=40_000)
        ref = means[ref_idx] + ref_rng.uniform(-1.0, 1.0, size=40_000)
        tri_batch: list[float] = []
        iid_batch: list[float] = []
        for replication in range(config.bounded_support_replications):
            tri_rng = np.random.default_rng(50_000 + 100 * n + replication)
            iid_rng = np.random.default_rng(60_000 + 100 * n + replication)
            tri_sample = means + tri_rng.uniform(-1.0, 1.0, size=n)
            iid_idx = iid_rng.integers(0, n, size=n)
            iid_sample = means[iid_idx] + iid_rng.uniform(-1.0, 1.0, size=n)
            tri_batch.append(w2_1d(tri_sample, ref))
            iid_batch.append(w2_1d(iid_sample, ref))
        tri_mean = float(np.mean(tri_batch))
        iid_mean = float(np.mean(iid_batch))
        tri_errors.append(tri_mean)
        iid_errors.append(iid_mean)
        curve_rows.extend(
            [
                {
                    "experiment": "bounded-support-fixed-span",
                    "setting": "triangular",
                    "n": n,
                    "value": round(tri_mean, 6),
                },
                {
                    "experiment": "bounded-support-fixed-span",
                    "setting": "iid-mixture",
                    "n": n,
                    "value": round(iid_mean, 6),
                },
            ]
        )

    tri_rate = fit_power_law_exponent(n_values, np.asarray(tri_errors, dtype=float))
    iid_rate = fit_power_law_exponent(n_values, np.asarray(iid_errors, dtype=float))
    rows.append(
        {
            "experiment": "bounded-support-fixed-span",
            "tri_rate_a": round(tri_rate, 6),
            "iid_rate_a": round(iid_rate, 6),
            "rate_gap": round(abs(tri_rate - iid_rate), 6),
            "span": config.fixed_span,
        }
    )
    return rows, curve_rows


def make_glue_research_figure(
    n_values: np.ndarray,
    tri_errors: np.ndarray,
    iid_errors: np.ndarray,
    c2_values: np.ndarray,
    span_values: np.ndarray,
    kappas: np.ndarray,
    betas: np.ndarray,
    predicted_error_exponents: np.ndarray,
    output_path: Path,
) -> None:
    figure = plt.figure(figsize=(16, 10), facecolor=BG)
    grid = figure.add_gridspec(2, 2, hspace=0.38, wspace=0.28)

    axis1 = figure.add_subplot(grid[0, 0])
    axis1.loglog(n_values, tri_errors, "o-", color=BLUE, lw=2, label="triangular")
    axis1.loglog(n_values, iid_errors, "s--", color=TEAL, lw=2, label="iid mixture")
    axis1.loglog(
        np.asarray([n_values[0], n_values[-1]], dtype=float),
        0.6 * np.asarray([n_values[0], n_values[-1]], dtype=float) ** (-0.5),
        ":",
        color=GREY,
        lw=1.2,
        label=r"$n^{-1/2}$ reference",
    )
    axis1.set_title("P2 fixed-span rate diagnostic")
    axis1.set_xlabel("n")
    axis1.set_ylabel(r"$E[W_2]$")
    axis1.legend(fontsize=8)

    axis2 = figure.add_subplot(grid[0, 1])
    axis2.semilogx(n_values, c2_values, "o-", color=PURPLE, lw=2)
    axis2.set_title(r"Asymptotic $C^2_{\mathrm{tri}}$ at fixed span")
    axis2.set_xlabel("n")
    axis2.set_ylabel(r"$C^2_{\mathrm{tri}}$")

    axis3 = figure.add_subplot(grid[1, 0])
    axis3.plot(span_values, kappas, "o-", color=ORANGE, lw=2)
    axis3.set_title(r"$\kappa(\Delta)$ for Gaussian window mixtures")
    axis3.set_xlabel(r"span $\Delta$")
    axis3.set_ylabel(r"$\kappa(\Delta)$")

    axis4 = figure.add_subplot(grid[1, 1])
    axis4.plot(betas, predicted_error_exponents, "o-", color=RED, lw=2)
    axis4.axhline(0.5, color=GREY, ls="--", lw=1.0, label=r"root-$n$")
    axis4.axhline(0.0, color=GREY, ls=":", lw=1.0)
    axis4.set_title("Predicted error exponent under span growth")
    axis4.set_xlabel(r"growth exponent $\beta$ in $\Delta_n \asymp n^\beta$")
    axis4.set_ylabel(r"predicted rate exponent")
    axis4.legend(fontsize=8)

    figure.suptitle("Glue theorem research diagnostics", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def run_glue_theorem_research(
    config: GlueTheoremResearchConfig | None = None,
) -> GlueTheoremResearchResult:
    if config is None:
        config = GlueTheoremResearchConfig()

    summary_rows: list[dict[str, str | float]] = []
    curve_rows: list[dict[str, str | float]] = []

    identity = kappa_identity_value()
    summary_rows.append(
        {
            "experiment": "identity-check",
            "integral_value": round(identity, 8),
            "homogeneous_kappa": round(float(np.sqrt(identity)), 8),
        }
    )

    (
        fixed_span_summary,
        fixed_span_curves,
        n_values,
        tri_errors,
        iid_errors,
        c2_values,
    ) = verify_fixed_span_bound(config)
    summary_rows.extend(fixed_span_summary)
    curve_rows.extend(fixed_span_curves)

    span_rows, span_values, kappas = kappa_vs_span_rows(config)
    summary_rows.extend(span_rows)
    curve_rows.extend(
        {
            "experiment": "kappa-vs-span",
            "setting": "kappa",
            "span": round(float(span), 6),
            "value": round(float(kappa), 6),
        }
        for span, kappa in zip(span_values, kappas, strict=True)
    )

    growth_rows, betas, predicted_error_exponents = span_growth_rows(config)
    summary_rows.extend(growth_rows)
    curve_rows.extend(
        {
            "experiment": "span-growth",
            "setting": "predicted-error-exponent",
            "beta": round(float(beta), 6),
            "value": round(float(exponent), 6),
        }
        for beta, exponent in zip(betas, predicted_error_exponents, strict=True)
    )

    bounded_rows, bounded_curves = bounded_support_inheritance_rows(config)
    summary_rows.extend(bounded_rows)
    curve_rows.extend(bounded_curves)

    return GlueTheoremResearchResult(summary_rows=summary_rows, curve_rows=curve_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P2 glue theorem research sweeps.")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("artifacts/csv/glue_theorem_research"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("artifacts/figures/glue_theorem_research"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GlueTheoremResearchConfig()
    result = run_glue_theorem_research(config)
    export_rows_csv(result.summary_rows, args.csv_dir / "glue_theorem_summary.csv")
    export_rows_csv(result.curve_rows, args.csv_dir / "glue_theorem_curves.csv")

    fixed_span_summary, _, n_values, tri_errors, iid_errors, c2_values = (
        verify_fixed_span_bound(config)
    )
    del fixed_span_summary
    _, span_values, kappas = kappa_vs_span_rows(config)
    _, betas, predicted_error_exponents = span_growth_rows(config)
    make_glue_research_figure(
        n_values=n_values,
        tri_errors=tri_errors,
        iid_errors=iid_errors,
        c2_values=c2_values,
        span_values=span_values,
        kappas=kappas,
        betas=betas,
        predicted_error_exponents=predicted_error_exponents,
        output_path=args.figures_dir / "glue_theorem_research.png",
    )


if __name__ == "__main__":
    main()
