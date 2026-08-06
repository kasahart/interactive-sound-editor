import numpy as np
from interactive_sound_editor.audio import ScipyAudioEngine, StftConfig, WandasAudioEngine

class Frame:
    def __init__(self, values, sample_rate=8000): self.values=np.asarray(values); self.sample_rate=sample_rate
    def __array__(self, dtype=None): return np.asarray(self.values, dtype=dtype)
    def stft(self, **kwargs):
        from scipy.signal import stft
        n=kwargs["win_length"]; hop=kwargs["hop_length"]
        _, _, z=stft(self.values, self.sample_rate, nperseg=n, noverlap=n-hop, nfft=kwargs["n_fft"], axis=-1)
        return SpectrumFrame(z, self.sample_rate, kwargs, self.values.shape[-1])
    def to_wav(self):
        from io import BytesIO
        from scipy.io import wavfile
        out=BytesIO(); wavfile.write(out, self.sample_rate, np.rint(np.asarray(self.values).T*32767).astype(np.int16)); return out

class SpectrumFrame(Frame):
    def __init__(self, values, sample_rate, config, length): super().__init__(values, sample_rate); self.config=config; self.length=length
    def __mul__(self, mask): return SpectrumFrame(self.values*mask, self.sample_rate, self.config, self.length)
    def istft(self):
        from scipy.signal import istft
        n=self.config["win_length"]; hop=self.config["hop_length"]
        _, x=istft(self.values, self.sample_rate, nperseg=n, noverlap=n-hop, nfft=self.config["n_fft"], time_axis=-1, freq_axis=-2)
        return Frame(x[..., :self.length], self.sample_rate)

class FakeWandas:
    @staticmethod
    def read(payload, file_type=None): return Frame(ScipyAudioEngine().load(payload).samples)
    @staticmethod
    def from_numpy(values, sample_rate): return Frame(values, sample_rate)


def test_public_wandas_pipeline_contract(wav_bytes):
    payload, channels, _, length = wav_bytes
    engine=WandasAudioEngine(FakeWandas); document=engine.load(payload)
    spectrum=engine.stft(document, StftConfig(128, 32, 128))
    output=engine.istft(engine.apply_mask(spectrum, np.ones(spectrum.values.shape)))
    assert output.samples.shape == (channels, length)
    assert np.isfinite(output.samples).all()
    assert engine.export_wav(output)[:4] == b"RIFF"
