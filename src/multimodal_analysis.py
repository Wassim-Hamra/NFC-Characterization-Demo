"""
multimodal_analysis.py
-----------------------
Gaussian Mixture Model (GMM) based detection of multimodal distributions.
Uses scikit-learn's GaussianMixture.
"""

import numpy as np
from dataclasses import dataclass, field
from sklearn.mixture import GaussianMixture


@dataclass
class GMMResult:
    n_components: int
    means: list[float]
    weights: list[float]
    stds: list[float]
    bic_1: float
    bic_2: float
    is_bimodal: bool
    bimodal_evidence: str          # human-readable strength
    component_labels: np.ndarray   # 0 or 1 per observation
    x_range: tuple[float, float]

    @property
    def bic_improvement(self) -> float:
        return self.bic_1 - self.bic_2   # positive = 2-component is better

    def insight_text(self, parameter_label: str, lot_label: str) -> str:
        """Generate a data-driven natural-language insight."""
        if not self.is_bimodal:
            return (
                f"The distribution of **{parameter_label}** in **{lot_label}** "
                f"appears unimodal. The GMM 2-component fit does not provide "
                f"a significantly better explanation than a single Gaussian "
                f"(ΔBIC = {self.bic_improvement:+.1f}). "
                f"No bimodal signal detected at this threshold."
            )

        m1, m2 = sorted(zip(self.means, self.weights), key=lambda x: x[0])
        separation = abs(self.means[1] - self.means[0])
        return (
            f"**Bimodal signal detected** in **{parameter_label}** for **{lot_label}**. "
            f"The GMM identifies two populations: "
            f"a lower group centred at **{m1[0]:.3f}** ({m1[1]*100:.1f}% of measurements) "
            f"and a higher group centred at **{m2[0]:.3f}** ({m2[1]*100:.1f}% of measurements), "
            f"with a separation of **{separation:.3f}** units. "
            f"The 2-component model is preferred over a single Gaussian by ΔBIC = {self.bic_improvement:.1f}. "
            f"This statistical pattern warrants further investigation — potential causes include "
            f"mixed process lots, test-site offsets, or a bimodal manufacturing parameter. "
            f"Note: a bimodal distribution is an investigation signal, not proof of a defect."
        )


def fit_gmm(
    values: np.ndarray,
    bic_threshold: float = 10.0,
) -> GMMResult:
    """
    Fit 1- and 2-component GMMs; determine whether the distribution is bimodal.

    Args:
        values: 1-D array of measurements (NaN removed by caller or here).
        bic_threshold: ΔBIC threshold to declare bimodality (default 10 = strong evidence).

    Returns:
        GMMResult with component statistics and bimodal flag.
    """
    values = values[~np.isnan(values)].reshape(-1, 1)

    if len(values) < 10:
        # Not enough data to fit GMM
        mean = float(np.mean(values))
        std  = float(np.std(values))
        return GMMResult(
            n_components=1,
            means=[mean], weights=[1.0], stds=[std],
            bic_1=0.0, bic_2=0.0,
            is_bimodal=False,
            bimodal_evidence="Insufficient data",
            component_labels=np.zeros(len(values), dtype=int),
            x_range=(float(values.min()), float(values.max())),
        )

    # Fit 1-component
    gmm1 = GaussianMixture(n_components=1, random_state=42, n_init=3)
    gmm1.fit(values)
    bic1 = float(gmm1.bic(values))

    # Fit 2-component
    gmm2 = GaussianMixture(n_components=2, random_state=42, n_init=5, covariance_type="full")
    gmm2.fit(values)
    bic2 = float(gmm2.bic(values))

    delta_bic = bic1 - bic2  # positive = 2-component wins
    is_bimodal = delta_bic > bic_threshold

    if is_bimodal:
        evidence = "Strong" if delta_bic > 50 else "Moderate"
        n_components = 2
        means = [float(m[0]) for m in gmm2.means_]
        weights = [float(w) for w in gmm2.weights_]
        stds = [float(np.sqrt(c[0][0])) for c in gmm2.covariances_]
        labels = gmm2.predict(values).astype(int)
    else:
        evidence = "None" if delta_bic < 2 else "Weak"
        n_components = 1
        means = [float(gmm1.means_[0][0])]
        weights = [1.0]
        stds = [float(np.sqrt(gmm1.covariances_[0][0][0]))]
        labels = np.zeros(len(values), dtype=int)

    return GMMResult(
        n_components=n_components,
        means=means,
        weights=weights,
        stds=stds,
        bic_1=bic1,
        bic_2=bic2,
        is_bimodal=is_bimodal,
        bimodal_evidence=evidence,
        component_labels=labels.flatten(),
        x_range=(float(values.min()), float(values.max())),
    )


def gmm_pdf(x: np.ndarray, result: GMMResult) -> np.ndarray:
    """Evaluate the full GMM mixture density at points x."""
    from scipy.stats import norm
    pdf = np.zeros_like(x, dtype=float)
    for mean, weight, std in zip(result.means, result.weights, result.stds):
        pdf += weight * norm.pdf(x, loc=mean, scale=std)
    return pdf


def component_pdf(x: np.ndarray, result: GMMResult, component_idx: int) -> np.ndarray:
    """Evaluate a single component's (scaled) density at points x."""
    from scipy.stats import norm
    mean  = result.means[component_idx]
    std   = result.stds[component_idx]
    weight = result.weights[component_idx]
    return weight * norm.pdf(x, loc=mean, scale=std)
