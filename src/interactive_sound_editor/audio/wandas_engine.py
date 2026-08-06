"""Minimal wandas-backed engine using only its documented public operations."""
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .types import AudioDocument, Spectrum, StftConfig


class WandasAudioEngine:
    """Adapter for wandas' read/from_numpy/STFT/ISTFT/WAV API.

    Import is delayed so the SciPy reference engine remains usable in Pyodide
    while a compatible wandas wheel is being produced.
    """
    def __init__(self, module: Any | None = None) -> None:
        if module is None:
            try:
                import wandas as module
            except ImportError as exc:
                raise RuntimeError("wandas is not installed; install the pinned dependency") from exc
        self._wd = module

    @staticmethod
    def _array(frame: Any) -> np.ndarray:
        # np.asarray is wandas' public NumPy interoperability boundary.
        return np.asarray(frame)

    @staticmethod
    def _sample_rate(frame: Any) -> int:
        for name in ("sample_rate", "sampling_rate"):
            value = getattr(frame, name, None)
            if value is not None:
                return int(value)
        raise TypeError("wandas ChannelFrame does not expose a public sample rate")

    def load(self, source: bytes | str | Path) -> AudioDocument:
        payload = str(source) if isinstance(source, Path) else source
        frame = self._wd.read(payload, file_type=".wav") if isinstance(payload, bytes) else self._wd.read(payload)
        values = self._array(frame)
        if values.ndim == 1:
            values = values[None, :]
        elif values.ndim == 2 and values.shape[0] > values.shape[1]:
            values = values.T
        return AudioDocument(np.nan_to_num(values.astype(float)), self._sample_rate(frame), frame)

    def stft(self, document: AudioDocument, config: StftConfig) -> Spectrum:
        frame = document.native
        if frame is None:
            frame = self._wd.from_numpy(document.samples, sample_rate=document.sample_rate)
        native = frame.stft(n_fft=config.n_fft, hop_length=config.hop_length,
                            win_length=config.win_length or config.n_fft, window=config.window)
        values = self._array(native)
        if values.ndim == 2:
            values = values[None, ...]
        frequencies = np.arange(values.shape[-2]) * document.sample_rate / config.n_fft
        times = np.arange(values.shape[-1]) * config.hop_length / document.sample_rate
        return Spectrum(values, frequencies, times, document.sample_rate, config,
                        document.samples.shape[-1], native)

    def apply_mask(self, spectrum: Spectrum, mask: np.ndarray) -> Spectrum:
        mask = np.broadcast_to(np.asarray(mask), spectrum.values.shape)
        native = spectrum.native * mask if spectrum.native is not None else None
        return replace(spectrum, values=np.nan_to_num(spectrum.values * mask), native=native)

    def istft(self, spectrum: Spectrum) -> AudioDocument:
        if spectrum.native is None:
            raise ValueError("wandas ISTFT requires a spectrum created by this engine")
        native = spectrum.native.istft()
        values = self._array(native)
        if values.ndim == 1:
            values = values[None, :]
        values = values[..., :spectrum.source_length]
        if values.shape[-1] < spectrum.source_length:
            values = np.pad(values, ((0, 0), (0, spectrum.source_length-values.shape[-1])))
        return AudioDocument(np.nan_to_num(values.astype(float)), spectrum.sample_rate, native)

    def export_wav(self, document: AudioDocument) -> bytes:
        frame = document.native if document.native is not None else self._wd.from_numpy(document.samples, sample_rate=document.sample_rate)
        result = frame.to_wav()
        return result.getvalue() if hasattr(result, "getvalue") else bytes(result)
