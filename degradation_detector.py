
import numpy as np
import cv2


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_image(image):

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
# FFT MAGNITUDE
# ============================================================

def get_fft_magnitude(image):

    image = normalize_image(image)

    fft = np.fft.fftshift(
        np.fft.fft2(image)
    )

    magnitude = np.abs(fft)

    return np.log1p(magnitude)


# ============================================================
# HIGH-FREQUENCY ENERGY
# ============================================================

def calculate_high_frequency_energy(image):

    magnitude = get_fft_magnitude(
        image
    )

    rows, cols = magnitude.shape

    cy = rows // 2
    cx = cols // 2

    Y, X = np.ogrid[
        :rows,
        :cols
    ]

    distance = np.sqrt(
        (X - cx) ** 2 +
        (Y - cy) ** 2
    )

    radius = min(
        rows,
        cols
    ) * 0.20

    high_frequency = (
        distance > radius
    )

    total_energy = np.sum(
        magnitude ** 2
    )

    high_energy = np.sum(
        magnitude[
            high_frequency
        ] ** 2
    )

    if total_energy < 1e-12:
        return 0.0

    return float(
        high_energy / total_energy
    )


# ============================================================
# DIRECTIONAL FREQUENCY ANALYSIS
# ============================================================

def calculate_directional_anisotropy(image):

    magnitude = get_fft_magnitude(
        image
    )

    rows, cols = magnitude.shape

    cy = rows // 2
    cx = cols // 2

    Y, X = np.indices(
        magnitude.shape
    )

    dx = X - cx
    dy = Y - cy

    angle = np.arctan2(
        dy,
        dx
    )

    # Focus on mid/high frequencies
    radius = np.sqrt(
        dx ** 2 + dy ** 2
    )

    mask = (
        radius > min(rows, cols) * 0.10
    ) & (
        radius < min(rows, cols) * 0.45
    )

    selected_angles = angle[mask]
    selected_energy = magnitude[mask]

    if len(selected_energy) == 0:
        return 0.0

    # Circular energy representation
    cos_component = np.sum(
        selected_energy *
        np.cos(2 * selected_angles)
    )

    sin_component = np.sum(
        selected_energy *
        np.sin(2 * selected_angles)
    )

    total = np.sum(
        selected_energy
    )

    if total < 1e-12:
        return 0.0

    anisotropy = np.sqrt(
        cos_component ** 2 +
        sin_component ** 2
    ) / total

    return float(
        np.clip(
            anisotropy,
            0,
            1
        )
    )


# ============================================================
# EDGE ORIENTATION ANALYSIS
# ============================================================

def calculate_edge_orientation_anisotropy(
    image
):

    image = normalize_image(
        image
    )

    image_uint8 = (
        image * 255
    ).astype(np.uint8)

    gx = cv2.Sobel(
        image_uint8,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image_uint8,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    magnitude = np.sqrt(
        gx ** 2 +
        gy ** 2
    )

    angle = np.arctan2(
        gy,
        gx
    )

    threshold = np.percentile(
        magnitude,
        85
    )

    mask = magnitude > threshold

    if np.sum(mask) < 10:
        return 0.0

    weights = magnitude[mask]
    angles = angle[mask]

    cos_component = np.sum(
        weights *
        np.cos(2 * angles)
    )

    sin_component = np.sum(
        weights *
        np.sin(2 * angles)
    )

    total = np.sum(
        weights
    )

    if total < 1e-12:
        return 0.0

    anisotropy = np.sqrt(
        cos_component ** 2 +
        sin_component ** 2
    ) / total

    return float(
        np.clip(
            anisotropy,
            0,
            1
        )
    )


# ============================================================
# BLUR INDICATOR
# ============================================================

def calculate_blur_indicator(image):

    image = normalize_image(
        image
    )

    image_uint8 = (
        image * 255
    ).astype(np.uint8)

    laplacian = cv2.Laplacian(
        image_uint8,
        cv2.CV_64F
    )

    variance = np.var(
        laplacian
    )

    # Higher variance generally means
    # stronger high-frequency detail.
    blur_indicator = 1.0 / (
        1.0 + variance / 100.0
    )

    return float(
        np.clip(
            blur_indicator,
            0,
            1
        )
    )


# ============================================================
# NOISE ESTIMATION
# ============================================================

def calculate_noise_indicator(image):

    image = normalize_image(
        image
    )

    blurred = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    residual = (
        image - blurred
    )

    noise_std = np.std(
        residual
    )

    return float(
        np.clip(
            noise_std * 10,
            0,
            1
        )
    )


# ============================================================
# DEGRADATION PROFILE
# ============================================================

def build_degradation_profile(
    image
):

    hf_energy = (
        calculate_high_frequency_energy(
            image
        )
    )

    frequency_anisotropy = (
        calculate_directional_anisotropy(
            image
        )
    )

    edge_anisotropy = (
        calculate_edge_orientation_anisotropy(
            image
        )
    )

    blur_indicator = (
        calculate_blur_indicator(
            image
        )
    )

    noise_indicator = (
        calculate_noise_indicator(
            image
        )
    )

    return {

        "high_frequency_energy":
            hf_energy,

        "frequency_anisotropy":
            frequency_anisotropy,

        "edge_anisotropy":
            edge_anisotropy,

        "blur_indicator":
            blur_indicator,

        "noise_indicator":
            noise_indicator
    }


# ============================================================
# INTELLIGENT CLASSIFICATION
# ============================================================

def classify_degradation(
    profile
):

    anisotropy = (
        profile["frequency_anisotropy"]
    )

    edge_anisotropy = (
        profile["edge_anisotropy"]
    )

    blur = (
        profile["blur_indicator"]
    )

    noise = (
        profile["noise_indicator"]
    )

    # --------------------------------------------------------
    # Motion blur
    # --------------------------------------------------------

    if (
        anisotropy > 0.20
        and edge_anisotropy > 0.20
    ):

        degradation = "Motion"

        confidence = min(
            1.0,
            (
                anisotropy +
                edge_anisotropy
            ) / 2
        )

        reason = (
            "Strong directional structure "
            "was detected in both frequency "
            "and edge domains."
        )

    # --------------------------------------------------------
    # Defocus blur
    # --------------------------------------------------------

    elif (
        blur > 0.45
        and anisotropy < 0.15
    ):

        degradation = "Defocus"

        confidence = min(
            1.0,
            blur * 1.2
        )

        reason = (
            "Blur appears relatively "
            "isotropic with reduced "
            "high-frequency detail."
        )

    # --------------------------------------------------------
    # Gaussian / isotropic blur
    # --------------------------------------------------------

    elif blur > 0.25:

        degradation = "Gaussian"

        confidence = min(
            1.0,
            0.55 + blur * 0.45
        )

        reason = (
            "Frequency characteristics suggest "
            "approximately isotropic smoothing."
        )

    # --------------------------------------------------------
    # No strong degradation
    # --------------------------------------------------------

    else:

        degradation = "Gaussian"

        confidence = 0.40

        reason = (
            "No strong directional degradation "
            "signature was detected; Gaussian "
            "is used as the conservative default."
        )

    # Noise is reported separately because
    # it can coexist with any blur type.

    return {

        "degradation":
            degradation,

        "confidence":
            float(
                np.clip(
                    confidence,
                    0,
                    1
                )
            ),

        "reason":
            reason,

        "noise_level":
            noise
    }


# ============================================================
# COMPLETE DETECTOR
# ============================================================

def detect_degradation(
    image
):

    profile = build_degradation_profile(
        image
    )

    classification = classify_degradation(
        profile
    )

    return {

        "profile":
            profile,

        "classification":
            classification
    }
