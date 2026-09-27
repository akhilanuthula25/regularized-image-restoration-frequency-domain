
import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


# ============================================================
# VALIDATE IMAGES
# ============================================================

def _normalize(image):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val < 1e-12:

        return np.zeros_like(image)

    return (
        image - min_val
    ) / (
        max_val - min_val
    )


# ============================================================
# GROUND-TRUTH METRICS
# ============================================================

def evaluate_against_ground_truth(
    ground_truth,
    restored
):

    ground_truth = _normalize(
        ground_truth
    )

    restored = _normalize(
        restored
    )

    mse = np.mean(
        (
            ground_truth -
            restored
        ) ** 2
    )

    rmse = np.sqrt(
        mse
    )

    psnr = peak_signal_noise_ratio(
        ground_truth,
        restored,
        data_range=1.0
    )

    ssim = structural_similarity(
        ground_truth,
        restored,
        data_range=1.0
    )

    return {

        "MSE":
            float(mse),

        "RMSE":
            float(rmse),

        "PSNR":
            float(psnr),

        "SSIM":
            float(ssim)
    }


# ============================================================
# BENCHMARK OPTIMIZATION
# ============================================================

def optimize_with_ground_truth(
    degraded_image,
    ground_truth,
    otf,
    method,
    restore_function,
    laplacian_otf=None,
    candidates=None
):

    if candidates is None:

        if method.lower() == "inverse":

            candidates = np.linspace(
                0.001,
                0.10,
                15
            )

        else:

            candidates = np.logspace(
                -6,
                -1,
                15
            )

    results = []

    best_parameter = None
    best_metrics = None
    best_image = None

    best_score = -np.inf

    for parameter in candidates:

        try:

            restored = restore_function(
                degraded_image,
                otf,
                method=method,
                parameter=float(parameter),
                laplacian_otf=laplacian_otf
            )

            metrics = evaluate_against_ground_truth(
                ground_truth,
                restored
            )

            # Combined benchmark score
            score = (
                0.70 *
                np.clip(
                    metrics["PSNR"] / 40.0,
                    0,
                    1
                )
                +
                0.30 *
                np.clip(
                    metrics["SSIM"],
                    0,
                    1
                )
            )

            results.append({

                "parameter":
                    float(parameter),

                "PSNR":
                    metrics["PSNR"],

                "SSIM":
                    metrics["SSIM"],

                "MSE":
                    metrics["MSE"],

                "score":
                    float(score)
            })

            if score > best_score:

                best_score = score

                best_parameter = float(
                    parameter
                )

                best_metrics = metrics

                best_image = restored

        except Exception:

            continue

    if best_parameter is None:

        raise RuntimeError(
            "Ground-truth optimization failed."
        )

    return {

        "mode":
            "benchmark",

        "best_parameter":
            best_parameter,

        "best_score":
            float(best_score),

        "best_metrics":
            best_metrics,

        "best_image":
            best_image,

        "results":
            results
    }
