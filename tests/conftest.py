from io import BytesIO
import numpy as np
import pytest
from scipy.io import wavfile

@pytest.fixture(params=[1, 2], ids=["mono", "stereo"])
def wav_bytes(request):
    sr = 8000
    t = np.arange(1600) / sr
    channels = [0.4 * np.sin(2*np.pi*440*t), 0.2 * np.sin(2*np.pi*660*t)]
    samples = np.stack(channels[:request.param], axis=1)
    if request.param == 1:
        samples = samples[:, 0]
    output = BytesIO()
    wavfile.write(output, sr, np.rint(samples*32767).astype(np.int16))
    return output.getvalue(), request.param, sr, len(t)
