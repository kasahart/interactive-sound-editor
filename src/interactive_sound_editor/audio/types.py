"""Value types shared by audio engines."""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StftConfig:
    n_fft: int = 1024
    hop_length: int = 512
    win_length: int | None = None
    window: str = "hann"

    def __post_init__(self) -> None:
        if self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("n_fft and hop_length must be positive")
        if self.win_length is not None and self.win_length <= 0:
            raise ValueError("win_length must be positive")


@dataclass(frozen=True)
class AudioDocument:
    """Normalized samples with the stable shape ``(channel, sample)``."""
    samples: NDArray[np.floating]
    sample_rate: int
    native: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim != 2:
            raise ValueError("samples must have shape (channel, sample)")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples must not contain NaN or Inf")


@dataclass(frozen=True)
class Spectrum:
    """Complex STFT with shape ``(channel, frequency, time)``."""
    values: NDArray[np.complexfloating]
    frequencies: NDArray[np.floating]
    times: NDArray[np.floating]
    sample_rate: int
    config: StftConfig
    source_length: int
    native: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 3:
            raise ValueError("spectrum must have shape (channel, frequency, time)")
        if values.shape[1:] != (len(self.frequencies), len(self.times)):
            raise ValueError("frequency/time axes do not match spectrum")
        if not np.all(np.isfinite(values)):
            raise ValueError("spectrum must not contain NaN or Inf")
