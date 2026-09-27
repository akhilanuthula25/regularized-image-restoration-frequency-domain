
import numpy as np
import cv2


# ============================================================
# BASIC UTILITIES
# ============================================================

def normalize_image(image):
    """
    Normalize image to float32 range [0, 1].
    """
    image = np.asarray(image, dtype=np.float32)

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val < 1e-12:
        return np.zeros_like(image, dtype=np.float32)

    return (image - min_val) / (max_val - min_val)


# ============================================================
# PSF -> OTF
# ============================================================

def psf_to_otf(psf, image_shape):
    """
    Convert a spatial-domain PSF into an OTF
    compatible with the requested image shape.
    """

    psf = np.asarray(psf, dtype=np.float32)

    if len(image_shape) != 2:
        raise ValueError("image_shape must be 2-dimensional.")

    otf = np.zeros(image_shape, dtype=np.float32)

    kh, kw = psf.shape

    if kh > image_shape[0] or kw > image_shape[1]:
        raise ValueError("PSF cannot be larger than image.")

    otf[:kh, :kw] = psf

    # Shift PSF center to the origin
    otf = np.roll(
        otf,
        -(kh // 2),
        axis=0
    )

    otf = np.roll(
        otf,
        -(kw // 2),
        axis=1
    )

    return np.fft.fft2(otf)


# ============================================================
# FFT / IFFT
# ============================================================

def compute_fft(image):
    """
    Compute centered 2D FFT.
    """
    return np.fft.fftshift(
        np.fft.fft2(image)
    )


def compute_ifft(frequency_data):
    """
    Reconstruct image from centered FFT.
    """
    return np.real(
        np.fft.ifft2(
            np.fft.ifftshift(frequency_data)
        )
    )


# ============================================================
# LAPLACIAN REGULARIZATION OPERATOR
# ============================================================

def create_laplacian_otf(image_shape):
    """
    Create frequency-domain Laplacian operator.
    """

    rows, cols = image_shape

    y = np.arange(rows) - rows // 2
    x = np.arange(cols) - cols // 2

    X, Y = np.meshgrid(x, y)

    laplacian = (
        -(X ** 2 + Y ** 2)
    ).astype(np.float32)

    laplacian = np.fft.ifftshift(laplacian)

    return np.fft.fft2(laplacian)


# ============================================================
# INVERSE FILTER
# ============================================================

def inverse_filter(
    degraded_image,
    otf,
    threshold=0.01
):
    """
    Basic inverse filtering with frequency threshold.
    """

    degraded_image = normalize_image(
        degraded_image
    )

    F = np.fft.fft2(degraded_image)

    H = otf

    H_abs = np.abs(H)

    mask = H_abs >= (
        threshold * np.max(H_abs)
    )

    restored_frequency = np.zeros_like(
        F,
        dtype=np.complex128
    )

    restored_frequency[mask] = (
        F[mask] / H[mask]
    )

    restored = np.real(
        np.fft.ifft2(restored_frequency)
    )

    return np.clip(
        restored,
        0,
        1
    )


# ============================================================
# WIENER FILTER
# ============================================================

def wiener_filter(
    degraded_image,
    otf,
    K=0.001
):
    """
    Wiener frequency-domain restoration.
    """

    degraded_image = normalize_image(
        degraded_image
    )

    F = np.fft.fft2(degraded_image)

    H = otf

    H_conj = np.conj(H)

    denominator = (
        np.abs(H) ** 2 + K
    )

    restored_frequency = (
        H_conj / denominator
    ) * F

    restored = np.real(
        np.fft.ifft2(restored_frequency)
    )

    return np.clip(
        restored,
        0,
        1
    )


# ============================================================
# TIKHONOV REGULARIZATION
# ============================================================

def tikhonov_restore(
    degraded_image,
    otf,
    laplacian_otf,
    lam=0.001
):
    """
    Tikhonov regularized frequency-domain restoration.
    """

    degraded_image = normalize_image(
        degraded_image
    )

    F = np.fft.fft2(degraded_image)

    H = otf
    L = laplacian_otf

    numerator = (
        np.conj(H) * F
    )

    denominator = (
        np.abs(H) ** 2
        + lam * np.abs(L) ** 2
        + 1e-12
    )

    restored_frequency = (
        numerator / denominator
    )

    restored = np.real(
        np.fft.ifft2(restored_frequency)
    )

    return np.clip(
        restored,
        0,
        1
    )


# ============================================================
# CONSTRAINED LEAST SQUARES
# ============================================================

def cls_restore(
    degraded_image,
    otf,
    laplacian_otf,
    gamma=0.001
):
    """
    Constrained Least Squares restoration.
    """

    degraded_image = normalize_image(
        degraded_image
    )

    F = np.fft.fft2(degraded_image)

    H = otf
    C = laplacian_otf

    numerator = (
        np.conj(H) * F
    )

    denominator = (
        np.abs(H) ** 2
        + gamma * np.abs(C) ** 2
        + 1e-12
    )

    restored_frequency = (
        numerator / denominator
    )

    restored = np.real(
        np.fft.ifft2(restored_frequency)
    )

    return np.clip(
        restored,
        0,
        1
    )


# ============================================================
# UNIFIED RESTORATION ENGINE
# ============================================================

def restore_image(
    degraded_image,
    otf,
    method="tikhonov",
    parameter=0.001,
    laplacian_otf=None
):
    """
    Unified restoration interface.
    """

    method = method.lower()

    if method == "inverse":

        return inverse_filter(
            degraded_image,
            otf,
            threshold=parameter
        )

    elif method == "wiener":

        return wiener_filter(
            degraded_image,
            otf,
            K=parameter
        )

    elif method == "tikhonov":

        if laplacian_otf is None:
            raise ValueError(
                "laplacian_otf is required "
                "for Tikhonov restoration."
            )

        return tikhonov_restore(
            degraded_image,
            otf,
            laplacian_otf,
            lam=parameter
        )

    elif method == "cls":

        if laplacian_otf is None:
            raise ValueError(
                "laplacian_otf is required "
                "for CLS restoration."
            )

        return cls_restore(
            degraded_image,
            otf,
            laplacian_otf,
            gamma=parameter
        )

    else:

        raise ValueError(
            f"Unknown restoration method: {method}"
        )
