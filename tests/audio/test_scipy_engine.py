from io import BytesIO
import numpy as np
from scipy.io import wavfile
from interactive_sound_editor.audio import ScipyAudioEngine, StftConfig


def test_wav_load_stft_mask_istft_export_round_trip(wav_bytes):
    payload, channels, sr, length = wav_bytes
    engine = ScipyAudioEngine()
    document = engine.load(payload)
    assert document.sample_rate == sr
    assert document.samples.shape == (channels, length)
    config = StftConfig(n_fft=256, hop_length=64, win_length=256)
    spectrum = engine.stft(document, config)
    assert spectrum.values.shape[:2] == (channels, 129)
    np.testing.assert_allclose(spectrum.frequencies, np.fft.rfftfreq(256, 1/sr))
    np.testing.assert_allclose(np.diff(spectrum.times), 64/sr)

    selected = engine.apply_mask(spectrum, np.ones(spectrum.values.shape[1:]))
    restored = engine.istft(selected)
    assert restored.samples.shape == document.samples.shape
    # Window overlap-add is numerically exact apart from floating point rounding.
    np.testing.assert_allclose(restored.samples, document.samples, atol=1e-10)
    exported = engine.export_wav(restored)
    rate, pcm = wavfile.read(BytesIO(exported))
    assert rate == sr and pcm.shape[0] == length
    assert np.isfinite(pcm).all()


def test_silence_tiny_and_short_audio_are_finite():
    engine = ScipyAudioEngine()
    for samples in (np.zeros(16), np.full(16, 1e-12)):
        output = BytesIO(); wavfile.write(output, 8000, samples.astype(np.float32))
        document = engine.load(output.getvalue())
        spectrum = engine.stft(document, StftConfig(n_fft=32, hop_length=8, win_length=16))
        restored = engine.istft(spectrum)
        assert np.isfinite(spectrum.values).all()
        assert np.isfinite(restored.samples).all()


def test_selected_plus_residual_equals_original(wav_bytes):
    payload, *_ = wav_bytes
    engine = ScipyAudioEngine(); spectrum = engine.stft(engine.load(payload), StftConfig(128, 32, 128))
    mask = np.linspace(0, 1, spectrum.values.shape[-1])[None, None, :]
    selected = engine.apply_mask(spectrum, mask)
    residual = engine.apply_mask(spectrum, 1-mask)
    np.testing.assert_allclose(selected.values + residual.values, spectrum.values, atol=1e-15)


def test_legacy_hop_length_was_actually_noverlap(wav_bytes):
    """Characterizes c_map: hop_length=64 yielded a 192-sample hop for n_fft=256."""
    from scipy.signal import stft
    payload, *_ = wav_bytes
    document = ScipyAudioEngine().load(payload)
    _, legacy_times, _ = stft(document.samples[0], document.sample_rate, nperseg=256, noverlap=64)
    assert np.diff(legacy_times)[0] == 192/document.sample_rate
