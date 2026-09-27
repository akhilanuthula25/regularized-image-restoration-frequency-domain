
import numpy as np


# ============================================================
# PARAMETER CANDIDATES
# ============================================================

def generate_parameter_candidates(method):

    method = method.lower()

    if method in ["wiener", "tikhonov", "cls"]:

        return np.logspace(
            -6,
            -1,
            15
        )

    elif method == "inverse":

        return np.linspace(
            0.001,
            0.10,
            15
        )

    else:

        raise ValueError(
            f"Unsupported restoration method: {method}"
        )


# ============================================================
# OPTIMIZATION SCORE
# ============================================================

def calculate_optimization_score(
    reference,
    restored
):

    reference = np.asarray(
        reference,
        dtype=np.float32
    )

    restored = np.asarray(
        restored,
        dtype=np.float32
    )

    reference = np.clip(
        reference,
        0,
        1
    )

    restored = np.clip(
        restored,
        0,
        1
    )

    mse = np.mean(
        (reference - restored) ** 2
    )

    if mse < 1e-12:

        psnr = 100.0

    else:

        psnr = 10 * np.log10(
            1.0 / mse
        )

    reference_centered = (
        reference - np.mean(reference)
    )

    restored_centered = (
        restored - np.mean(restored)
    )

    denominator = (
        np.sqrt(
            np.sum(
                reference_centered ** 2
            )
        )
        *
        np.sqrt(
            np.sum(
                restored_centered ** 2
            )
        )
        + 1e-12
    )

    correlation = (
        np.sum(
            reference_centered *
            restored_centered
        )
        / denominator
    )

    correlation = np.clip(
        correlation,
        -1,
        1
    )

    psnr_score = np.clip(
        psnr / 40.0,
        0,
        1
    )

    correlation_score = (
        correlation + 1
    ) / 2

    score = (
        0.65 * psnr_score +
        0.35 * correlation_score
    )

    return float(score)


# ============================================================
# PARAMETER OPTIMIZATION
# ============================================================

def optimize_restoration_parameter(
    degraded_image,
    otf,
    method,
    restore_function,
    laplacian_otf=None,
    candidates=None
):

    method = method.lower()

    if candidates is None:

        candidates = (
            generate_parameter_candidates(
                method
            )
        )

    results = []

    best_parameter = None
    best_score = -np.inf
    best_image = None

    for parameter in candidates:

        try:

            restored = restore_function(
                degraded_image,
                otf,
                method=method,
                parameter=float(parameter),
                laplacian_otf=laplacian_otf
            )

            score = calculate_optimization_score(
                degraded_image,
                restored
            )

            results.append({
                "parameter": float(parameter),
                "score": float(score)
            })

            if score > best_score:

                best_score = score
                best_parameter = float(
                    parameter
                )
                best_image = restored

        except Exception:

            continue

    if best_parameter is None:

        raise RuntimeError(
            "No valid restoration parameter "
            "could be evaluated."
        )

    return {
        "best_parameter": best_parameter,
        "best_score": float(best_score),
        "best_image": best_image,
        "results": results
    }
