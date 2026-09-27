
import numpy as np


# ============================================================
# L-CURVE PARAMETER CANDIDATES
# ============================================================

def generate_lcurve_candidates():

    return np.logspace(
        -6,
        -1,
        20
    )


# ============================================================
# L-CURVE ANALYSIS
# ============================================================

def calculate_lcurve_point(
    degraded_image,
    restored,
    otf,
    laplacian_otf
):

    degraded_image = np.asarray(
        degraded_image,
        dtype=np.float32
    )

    restored = np.asarray(
        restored,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Data fidelity residual
    # ||Hx - y||
    # --------------------------------------------------------

    restored_frequency = np.fft.fft2(
        restored
    )

    predicted_frequency = (
        otf *
        restored_frequency
    )

    predicted = np.real(
        np.fft.ifft2(
            predicted_frequency
        )
    )

    residual_norm = np.linalg.norm(
        degraded_image - predicted
    )

    # --------------------------------------------------------
    # Regularization norm
    # ||Lx||
    # --------------------------------------------------------

    regularized_frequency = (
        laplacian_otf *
        restored_frequency
    )

    regularized = np.real(
        np.fft.ifft2(
            regularized_frequency
        )
    )

    regularization_norm = np.linalg.norm(
        regularized
    )

    return (
        float(residual_norm),
        float(regularization_norm)
    )


# ============================================================
# CURVATURE ESTIMATION
# ============================================================

def estimate_lcurve_corner(
    parameters,
    residuals,
    regularizations
):

    parameters = np.asarray(
        parameters,
        dtype=np.float64
    )

    residuals = np.asarray(
        residuals,
        dtype=np.float64
    )

    regularizations = np.asarray(
        regularizations,
        dtype=np.float64
    )

    x = np.log10(
        residuals + 1e-12
    )

    y = np.log10(
        regularizations + 1e-12
    )

    t = np.log10(
        parameters
    )

    if len(parameters) < 3:

        return int(
            len(parameters) // 2
        )

    dx = np.gradient(
        x,
        t
    )

    dy = np.gradient(
        y,
        t
    )

    ddx = np.gradient(
        dx,
        t
    )

    ddy = np.gradient(
        dy,
        t
    )

    denominator = (
        dx ** 2 +
        dy ** 2
    ) ** 1.5 + 1e-12

    curvature = np.abs(
        dx * ddy -
        dy * ddx
    ) / denominator

    # Ignore unstable boundary points
    curvature[:1] = -np.inf
    curvature[-1:] = -np.inf

    corner_index = int(
        np.argmax(curvature)
    )

    return corner_index


# ============================================================
# AUTOMATIC L-CURVE OPTIMIZATION
# ============================================================

def optimize_lcurve(
    degraded_image,
    otf,
    restore_function,
    method="tikhonov",
    laplacian_otf=None,
    candidates=None
):

    method = method.lower()

    if candidates is None:

        candidates = (
            generate_lcurve_candidates()
        )

    residuals = []
    regularizations = []
    restorations = []

    valid_parameters = []

    for parameter in candidates:

        try:

            restored = restore_function(
                degraded_image,
                otf,
                method=method,
                parameter=float(parameter),
                laplacian_otf=laplacian_otf
            )

            residual_norm, regularization_norm = (
                calculate_lcurve_point(
                    degraded_image,
                    restored,
                    otf,
                    laplacian_otf
                )
            )

            residuals.append(
                residual_norm
            )

            regularizations.append(
                regularization_norm
            )

            restorations.append(
                restored
            )

            valid_parameters.append(
                float(parameter)
            )

        except Exception:

            continue

    if len(valid_parameters) < 3:

        raise RuntimeError(
            "Insufficient valid points "
            "for L-curve optimization."
        )

    corner_index = estimate_lcurve_corner(
        valid_parameters,
        residuals,
        regularizations
    )

    best_parameter = (
        valid_parameters[
            corner_index
        ]
    )

    best_image = (
        restorations[
            corner_index
        ]
    )

    return {

        "mode":
            "deployment_lcurve",

        "best_parameter":
            float(best_parameter),

        "best_image":
            best_image,

        "parameters":
            valid_parameters,

        "residuals":
            residuals,

        "regularizations":
            regularizations,

        "corner_index":
            int(corner_index)
    }
