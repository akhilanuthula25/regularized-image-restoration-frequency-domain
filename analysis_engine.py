
import numpy as np
import cv2

from scipy.ndimage import gaussian_filter


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
# NOISE ESTIMATION
# ============================================================

def estimate_noise_mad(image):

    image = normalize_image(image)

    median = np.median(image)

    deviation = np.abs(
        image - median
    )

    mad = np.median(deviation)

    sigma = (
        1.4826 * mad
    )

    return float(sigma)


# ============================================================
# FREQUENCY ENERGY
# ============================================================

def calculate_frequency_energy_ratio(image):

    image = normalize_image(image)

    fft = np.fft.fftshift(
        np.fft.fft2(image)
    )

    magnitude = np.abs(fft)

    rows, cols = image.shape

    cy = rows // 2
    cx = cols // 2

    radius = min(rows, cols) * 0.15

    Y, X = np.ogrid[
        :rows,
        :cols
    ]

    distance = np.sqrt(
        (X - cx) ** 2 +
        (Y - cy) ** 2
    )

    high_frequency_mask = (
        distance > radius
    )

    total_energy = np.sum(
        magnitude ** 2
    )

    high_frequency_energy = np.sum(
        magnitude[high_frequency_mask] ** 2
    )

    if total_energy < 1e-12:
        return 0.0

    return float(
        high_frequency_energy /
        total_energy
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image):

    image = normalize_image(image)

    noise_level = estimate_noise_mad(
        image
    )

    hf_ratio = calculate_frequency_energy_ratio(
        image
    )

    mean_intensity = float(
        np.mean(image)
    )

    std_intensity = float(
        np.std(image)
    )

    return {
        "noise_level": noise_level,
        "high_frequency_energy_ratio": hf_ratio,
        "mean_intensity": mean_intensity,
        "intensity_std": std_intensity
    }


# ============================================================
# DEGRADATION SEVERITY
# ============================================================

def calculate_degradation_severity(
    analysis
):

    noise = analysis.get(
        "noise_level",
        0.0
    )

    hf_ratio = analysis.get(
        "high_frequency_energy_ratio",
        0.0
    )

    # Noise contribution
    noise_score = min(
        noise / 0.15,
        1.0
    )

    # Lower high-frequency energy
    # indicates stronger blur/smoothing.
    hf_score = 1.0 - min(
        hf_ratio / 0.40,
        1.0
    )

    severity = (
        0.5 * noise_score +
        0.5 * hf_score
    )

    return float(
        np.clip(
            severity,
            0,
            1
        )
    )


# ============================================================
# RESTORATION METHOD RECOMMENDATION
# ============================================================

def recommend_method(
    analysis,
    severity=None
):

    if severity is None:
        severity = calculate_degradation_severity(
            analysis
        )

    noise = analysis.get(
        "noise_level",
        0.0
    )

    if noise > 0.08:

        method = "wiener"

        reason = (
            "Higher estimated noise detected; "
            "Wiener filtering provides "
            "noise-aware regularization."
        )

    elif severity > 0.65:

        method = "tikhonov"

        reason = (
            "Strong degradation detected; "
            "Tikhonov regularization provides "
            "stable frequency-domain restoration."
        )

    elif severity > 0.35:

        method = "cls"

        reason = (
            "Moderate degradation detected; "
            "CLS provides controlled regularization."
        )

    else:

        method = "inverse"

        reason = (
            "Relatively mild degradation detected; "
            "inverse filtering can recover "
            "frequency information."
        )

    return {
        "method": method,
        "reason": reason,
        "severity": float(severity)
    }


# ============================================================
# EDGE MAP
# ============================================================

def get_edge_map(image):

    image = normalize_image(image)

    image_uint8 = (
        image * 255
    ).astype(np.uint8)

    edges = cv2.Canny(
        image_uint8,
        50,
        150
    )

    return edges


# ============================================================
# EDGE PRESERVATION
# ============================================================

def calculate_edge_preservation(
    reference,
    restored
):

    reference_edges = get_edge_map(
        reference
    )

    restored_edges = get_edge_map(
        restored
    )

    ref_count = np.sum(
        reference_edges > 0
    )

    if ref_count == 0:
        return 0.0

    overlap = np.sum(
        (
            reference_edges > 0
        ) &
        (
            restored_edges > 0
        )
    )

    return float(
        overlap / ref_count
    )


# ============================================================
# RINGING INDICATOR
# ============================================================

def calculate_ringing_indicator(
    restored
):

    image = normalize_image(
        restored
    )

    blurred = gaussian_filter(
        image,
        sigma=1.0
    )

    high_frequency = (
        image - blurred
    )

    positive = np.sum(
        high_frequency > 0.08
    )

    negative = np.sum(
        high_frequency < -0.08
    )

    total = image.size

    indicator = (
        positive + negative
    ) / max(total, 1)

    return float(
        np.clip(
            indicator * 10,
            0,
            1
        )
    )


# ============================================================
# SMOOTHING INDICATOR
# ============================================================

def calculate_smoothing_indicator(
    original,
    restored
):

    original = normalize_image(
        original
    )

    restored = normalize_image(
        restored
    )

    original_std = np.std(
        original
    )

    restored_std = np.std(
        restored
    )

    if original_std < 1e-12:
        return 0.0

    reduction = (
        original_std - restored_std
    ) / original_std

    return float(
        np.clip(
            reduction,
            0,
            1
        )
    )


# ============================================================
# RESTORATION CONFIDENCE
# ============================================================

def calculate_restoration_confidence(
    psnr=None,
    ssim=None,
    edge_preservation=0.0,
    ringing=0.0
):

    components = []

    if psnr is not None:

        psnr_score = np.clip(
            float(psnr) / 40.0,
            0,
            1
        )

        components.append(
            psnr_score
        )

    if ssim is not None:

        components.append(
            np.clip(
                float(ssim),
                0,
                1
            )
        )

    components.append(
        np.clip(
            edge_preservation,
            0,
            1
        )
    )

    components.append(
        1.0 - np.clip(
            ringing,
            0,
            1
        )
    )

    if not components:
        return 0.0

    confidence = np.mean(
        components
    )

    return float(
        np.clip(
            confidence,
            0,
            1
        )
    )


# ============================================================
# HEALTH SCORE
# ============================================================

def calculate_health_score(
    confidence,
    edge_preservation,
    ringing,
    smoothing
):

    score = (
        0.40 * confidence +
        0.30 * edge_preservation +
        0.15 * (1 - ringing) +
        0.15 * (1 - smoothing)
    )

    return float(
        np.clip(
            score * 100,
            0,
            100
        )
    )


# ============================================================
# QUALITY WARNINGS
# ============================================================

def generate_quality_warnings(
    ringing,
    smoothing,
    edge_preservation,
    confidence
):

    warnings = []

    if ringing > 0.35:

        warnings.append(
            "Potential ringing artifacts detected."
        )

    if smoothing > 0.40:

        warnings.append(
            "Potential over-smoothing detected."
        )

    if edge_preservation < 0.50:

        warnings.append(
            "Edge preservation is relatively low."
        )

    if confidence < 0.50:

        warnings.append(
            "Restoration confidence is relatively low."
        )

    if not warnings:

        warnings.append(
            "No major restoration quality warnings detected."
        )

    return warnings
