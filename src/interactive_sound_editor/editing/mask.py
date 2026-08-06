"""Pure NumPy mask calculations extracted from the UI module."""
import numpy as np
from scipy.ndimage import median_filter


def softmask(x: np.ndarray, y: np.ndarray, power: float = 2.0) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = x / (x + y)
    return np.nan_to_num(np.clip(result, 0, 1)) ** power


def harmonic_mask(spectrum: np.ndarray, kernel_size: int = 31, power: float = 2.0, margin: float = 1.0) -> np.ndarray:
    magnitude = np.abs(spectrum)
    harmonic = median_filter(magnitude, size=[1] * (magnitude.ndim - 1) + [kernel_size], mode="reflect")
    percussive = median_filter(magnitude, size=[1] * (magnitude.ndim - 2) + [kernel_size, 1], mode="reflect")
    return softmask(harmonic, percussive * margin, power)


def percussive_mask(spectrum: np.ndarray, kernel_size: int = 31, power: float = 2.0, margin: float = 1.0) -> np.ndarray:
    magnitude = np.abs(spectrum)
    harmonic = median_filter(magnitude, size=[1] * (magnitude.ndim - 1) + [kernel_size], mode="reflect")
    percussive = median_filter(magnitude, size=[1] * (magnitude.ndim - 2) + [kernel_size, 1], mode="reflect")
    return softmask(percussive, harmonic * margin, power)


def zscore_mask(values: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    magnitude = np.abs(values)
    std = magnitude.std()
    if std == 0 or not np.isfinite(std):
        return np.zeros_like(magnitude, dtype=float)
    return (np.abs((magnitude - magnitude.mean()) / std) >= threshold).astype(float)


def frequency_trend_mask(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(values)
    trend = magnitude.mean(axis=-1, keepdims=True)
    return np.divide(magnitude, trend, out=np.zeros_like(magnitude, dtype=float), where=trend != 0)


def db_attenuation(mask: np.ndarray, reduction_db: float) -> np.ndarray:
    return np.where(np.asarray(mask, dtype=bool), 10.0 ** (-reduction_db / 20.0), 1.0)


def split_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = np.clip(np.nan_to_num(mask), 0.0, 1.0)
    return selected, 1.0 - selected
