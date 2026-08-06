"""AudioEngine preserving the editor's SciPy conventions where practical."""
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import istft as scipy_istft
from scipy.signal import stft as scipy_stft

from .types import AudioDocument, Spectrum, StftConfig


def _normalized(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.integer):
        scale = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
        result = data.astype(np.float64) / scale
    else:
        result = data.astype(np.float64)
    return np.nan_to_num(result, copy=False)


class ScipyAudioEngine:
    """SciPy reference engine using hop_length in its conventional meaning."""
    def load(self, source: bytes | str | Path) -> AudioDocument:
        sample_rate, data = wavfile.read(BytesIO(source) if isinstance(source, bytes) else source)
        samples = _normalized(data)
        if samples.ndim == 1:
            samples = samples[:, None]
        return AudioDocument(np.ascontiguousarray(samples.T), int(sample_rate))

    def stft(self, document: AudioDocument, config: StftConfig) -> Spectrum:
        win_length = config.win_length or config.n_fft
        noverlap = win_length - config.hop_length
        if not 0 <= noverlap < win_length:
            raise ValueError("hop_length must be between 1 and win_length")
        freqs, times, values = scipy_stft(
            document.samples, fs=document.sample_rate, window=config.window,
            nperseg=win_length, noverlap=noverlap, nfft=config.n_fft,
            axis=-1, boundary="zeros", padded=True,
        )
        return Spectrum(values, freqs, times, document.sample_rate, config, document.samples.shape[-1])

    def apply_mask(self, spectrum: Spectrum, mask: np.ndarray) -> Spectrum:
        mask = np.asarray(mask)
        try:
            values = spectrum.values * np.broadcast_to(mask, spectrum.values.shape)
        except ValueError as exc:
            raise ValueError("mask is not broadcastable to (channel, frequency, time)") from exc
        return replace(spectrum, values=np.nan_to_num(values))

    def istft(self, spectrum: Spectrum) -> AudioDocument:
        win_length = spectrum.config.win_length or spectrum.config.n_fft
        _, samples = scipy_istft(
            spectrum.values, fs=spectrum.sample_rate, window=spectrum.config.window,
            nperseg=win_length, noverlap=win_length-spectrum.config.hop_length,
            nfft=spectrum.config.n_fft, input_onesided=True, boundary=True,
            time_axis=-1, freq_axis=-2,
        )
        length = spectrum.source_length
        samples = samples[..., :length]
        if samples.shape[-1] < length:
            samples = np.pad(samples, ((0, 0), (0, length-samples.shape[-1])))
        return AudioDocument(np.nan_to_num(samples), spectrum.sample_rate)

    def export_wav(self, document: AudioDocument) -> bytes:
        output = BytesIO()
        samples = np.clip(document.samples, -1.0, 1.0)
        pcm = np.rint(samples.T * 32767.0).astype(np.int16)
        wavfile.write(output, document.sample_rate, pcm)
        return output.getvalue()
