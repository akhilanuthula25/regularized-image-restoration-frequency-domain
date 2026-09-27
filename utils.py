
import numpy as np
import cv2
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)


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


def convert_to_grayscale(image):

    if image is None:
        raise ValueError("Image is None.")

    if len(image.shape) == 2:
        return image

    if image.shape[2] == 4:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGBA2RGB
        )

    return cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )


def resize_image(
    image,
    size=(512, 512)
):

    return cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA
    )


def validate_image(image):

    if image is None:
        return False

    if image.size == 0:
        return False

    if len(image.shape) not in [2, 3]:
        return False

    return True


def calculate_metrics(
    reference,
    restored
):

    reference = normalize_image(
        reference
    )

    restored = normalize_image(
        restored
    )

    mse = np.mean(
        (reference - restored) ** 2
    )

    rmse = np.sqrt(mse)

    psnr = peak_signal_noise_ratio(
        reference,
        restored,
        data_range=1.0
    )

    ssim = structural_similarity(
        reference,
        restored,
        data_range=1.0
    )

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "PSNR": float(psnr),
        "SSIM": float(ssim)
    }
