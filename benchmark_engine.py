
import numpy as np
import pandas as pd

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

from restoration_core import (
    inverse_filter,
    wiener_filter,
    tikhonov_restore,
    cls_restore
)

from analysis_engine import calculate_edge_preservation


def normalize_image(image):
    image = np.asarray(image, dtype=np.float32)

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val < 1e-8:
        return np.zeros_like(image)

    return (image - min_val) / (max_val - min_val)


def calculate_basic_metrics(reference, restored):
    """
    Calculate standard image restoration metrics.
    """

    reference = normalize_image(reference)
    restored = normalize_image(restored)

    mse = float(np.mean((reference - restored) ** 2))
    rmse = float(np.sqrt(mse))

    psnr = float(
        peak_signal_noise_ratio(
            reference,
            restored,
            data_range=1.0
        )
    )

    ssim = float(
        structural_similarity(
            reference,
            restored,
            data_range=1.0
        )
    )

    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "MSE": mse,
        "RMSE": rmse
    }


def calculate_edge_metric(reference, restored):
    """
    Calculate edge preservation.
    """

    try:
        value = calculate_edge_preservation(
            reference,
            restored
        )

        if isinstance(value, dict):
            for key in [
                "edge_preservation",
                "score",
                "value"
            ]:
                if key in value:
                    return float(value[key])

        return float(value)

    except Exception:
        return np.nan


def calculate_composite_score(metrics):
    """
    Composite benchmark score.

    Higher is better.

    The score combines:
    - normalized PSNR
    - SSIM
    - edge preservation
    """

    psnr = metrics["PSNR"]

    # Practical bounded normalization.
    psnr_score = np.clip((psnr - 10.0) / 40.0, 0.0, 1.0)

    ssim_score = np.clip(metrics["SSIM"], 0.0, 1.0)

    edge_score = metrics.get(
        "Edge Preservation",
        np.nan
    )

    if np.isnan(edge_score):
        edge_score = 0.0

    edge_score = np.clip(edge_score, 0.0, 1.0)

    score = (
        0.45 * psnr_score +
        0.40 * ssim_score +
        0.15 * edge_score
    )

    return float(score * 100.0)


def benchmark_restoration_methods(
    clean_image,
    degraded_image,
    otf,
    laplacian_otf,
    parameters=None
):
    """
    Benchmark all supported restoration methods
    against a clean ground-truth image.
    """

    if parameters is None:
        parameters = {
            "inverse": 0.01,
            "wiener": 0.001,
            "tikhonov": 0.001,
            "cls": 0.001
        }

    results = []

    # -------------------------
    # Inverse
    # -------------------------

    inverse_result = inverse_filter(
        degraded_image,
        otf,
        threshold=parameters["inverse"]
    )

    results.append(
        _evaluate_method(
            "Inverse",
            clean_image,
            inverse_result
        )
    )

    # -------------------------
    # Wiener
    # -------------------------

    wiener_result = wiener_filter(
        degraded_image,
        otf,
        K=parameters["wiener"]
    )

    results.append(
        _evaluate_method(
            "Wiener",
            clean_image,
            wiener_result
        )
    )

    # -------------------------
    # Tikhonov
    # -------------------------

    tikhonov_result = tikhonov_restore(
        degraded_image,
        otf,
        laplacian_otf,
        lam=parameters["tikhonov"]
    )

    results.append(
        _evaluate_method(
            "Tikhonov",
            clean_image,
            tikhonov_result
        )
    )

    # -------------------------
    # CLS
    # -------------------------

    cls_result = cls_restore(
        degraded_image,
        otf,
        laplacian_otf,
        gamma=parameters["cls"]
    )

    results.append(
        _evaluate_method(
            "CLS",
            clean_image,
            cls_result
        )
    )

    df = pd.DataFrame(results)

    return df


def _evaluate_method(method_name, clean_image, restored_image):

    metrics = calculate_basic_metrics(
        clean_image,
        restored_image
    )

    metrics["Edge Preservation"] = calculate_edge_metric(
        clean_image,
        restored_image
    )

    metrics["Method"] = method_name

    metrics["Composite Score"] = calculate_composite_score(
        metrics
    )

    return metrics


def rank_methods(benchmark_df):
    """
    Rank restoration methods using composite benchmark score.
    """

    df = benchmark_df.copy()

    df = df.sort_values(
        "Composite Score",
        ascending=False
    ).reset_index(drop=True)

    df["Benchmark Rank"] = np.arange(
        1,
        len(df) + 1
    )

    return df


def get_benchmark_summary(benchmark_df):

    ranked = rank_methods(benchmark_df)

    best_row = ranked.iloc[0]

    return {
        "best_method": str(best_row["Method"]),
        "best_score": float(
            best_row["Composite Score"]
        ),
        "best_psnr": float(
            best_row["PSNR"]
        ),
        "best_ssim": float(
            best_row["SSIM"]
        ),
        "ranking": ranked[
            [
                "Benchmark Rank",
                "Method",
                "PSNR",
                "SSIM",
                "Composite Score"
            ]
        ].to_dict("records")
    }
