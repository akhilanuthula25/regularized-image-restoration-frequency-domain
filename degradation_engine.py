
import numpy as np
import cv2

EPSILON = 1e-8


def normalize_psf(psf):
    psf = np.asarray(psf, dtype=np.float32)

    total = np.sum(psf)

    if total < EPSILON:
        raise ValueError("PSF energy is zero.")

    return psf / total


def create_gaussian_psf(size=15, sigma=3.0):

    if size % 2 == 0:
        size += 1

    kernel_1d = cv2.getGaussianKernel(
        size,
        sigma
    )

    psf = kernel_1d @ kernel_1d.T

    return normalize_psf(psf)


def create_motion_psf(size=15, angle=0):

    if size % 2 == 0:
        size += 1

    psf = np.zeros(
        (size, size),
        dtype=np.float32
    )

    center = size // 2

    psf[center, :] = 1.0

    rotation_matrix = cv2.getRotationMatrix2D(
        (center, center),
        angle,
        1.0
    )

    psf = cv2.warpAffine(
        psf,
        rotation_matrix,
        (size, size)
    )

    return normalize_psf(psf)


def create_defocus_psf(size=15):

    if size % 2 == 0:
        size += 1

    psf = np.zeros(
        (size, size),
        dtype=np.float32
    )

    center = size // 2

    cv2.circle(
        psf,
        (center, center),
        center,
        1,
        -1
    )

    return normalize_psf(psf)


def create_custom_psf(psf_array):

    psf = np.asarray(
        psf_array,
        dtype=np.float32
    )

    if psf.ndim != 2:
        raise ValueError(
            "PSF must be a 2D matrix."
        )

    if np.sum(np.abs(psf)) < EPSILON:
        raise ValueError(
            "Custom PSF cannot contain only zeros."
        )

    return normalize_psf(psf)


def psf_to_otf(psf, image_shape):

    psf = np.asarray(
        psf,
        dtype=np.float32
    )

    h, w = image_shape
    kh, kw = psf.shape

    if kh > h or kw > w:
        raise ValueError(
            "PSF cannot be larger than the image."
        )

    otf = np.zeros(
        image_shape,
        dtype=np.float32
    )

    otf[:kh, :kw] = psf

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


def get_frequency_response(otf):

    magnitude = np.abs(otf)

    maximum = np.max(magnitude)

    if maximum > EPSILON:
        magnitude = magnitude / maximum

    return np.fft.fftshift(magnitude)


def create_psf(
    model,
    size=15,
    sigma=3.0,
    angle=0
):

    model = model.lower()

    if model == "gaussian":

        return create_gaussian_psf(
            size=size,
            sigma=sigma
        )

    elif model == "motion":

        return create_motion_psf(
            size=size,
            angle=angle
        )

    elif model == "defocus":

        return create_defocus_psf(
            size=size
        )

    else:

        raise ValueError(
            f"Unsupported degradation model: {model}"
        )


def analyze_psf(psf):

    psf = normalize_psf(psf)

    return {
        "height": int(psf.shape[0]),
        "width": int(psf.shape[1]),
        "sum": float(np.sum(psf)),
        "maximum": float(np.max(psf)),
        "mean": float(np.mean(psf)),
        "nonzero_ratio": float(
            np.mean(psf > 0)
        )
    }
