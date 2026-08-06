# wandas engine spike: investigation and migration plan

## Current responsibilities and data flow

`c_map.py` combines seven layers. `AudioModel` reads/writes WAV through
`scipy.io.wavfile`, normalizes samples, computes SciPy STFT/ISTFT, applies masks,
and contains the five-iteration legacy Griffin–Lim reconstruction. `AudioViewModel`
owns selection indices, mask algorithms and model-list undo/redo. `AudioView` owns
ipywidgets, Matplotlib drawing and its 500 ms timer. `InteractiveSoundEditor` owns
file upload and object creation. `cmap_pysc.py` subclasses those UI classes to
replace widgets with Panel and connects `js.document` browser events.
`pyscript.toml` declares Pyodide packages/files; `index.html` pins the PyScript,
Bokeh and Panel browser assets and supplies their DOM mount points.

The legacy path is `wavfile.read` -> division by the input peak -> `scipy.signal.stft`
-> mask multiplication -> five alternating ISTFT/STFT phase updates -> inverse
scaling/cast -> `wavfile.write`. Notably, `AudioModel.hop_length` is passed as
SciPy `noverlap`; with `n_fft=1024` and the default value 512 this happens to equal
a 512-sample hop, but non-default values mean `actual hop = win_length -
hop_length`. The characterization test records this before the new engines give
`hop_length` its standard wandas meaning.

## wandas investigation

The requested public boundary consists of `wandas.read`, `wandas.from_numpy`,
`ChannelFrame.stft`, `SpectrogramFrame.istft`, frame/NumPy multiplication and
`ChannelFrame.to_wav`. The adapter uses only these operations plus public NumPy
conversion and public `sample_rate`/`sampling_rate` metadata; it does not inspect
private attributes. A live checkout/package inspection could not be completed in
this environment because GitHub and the Python index returned HTTP proxy 403 and
wandas was not preinstalled. Therefore the injected-module contract test validates
the public call sequence, but a pinned revision and real-package comparison must
be finalized when the repository is reachable. No unverified wandas version was
invented or added to the runtime lock.

The existing Pyodide configuration does not include wandas. Pyodide compatibility
is consequently **unverified**: a pure-Python wheel (or Pyodide-compatible wheels
for every compiled dependency), bytes I/O, and `to_wav` behavior must be checked
in a browser. Existing browser versions remain unchanged in this spike.

## Adopted design

The new UI-free package establishes `AudioEngine`, immutable `AudioDocument`
(channel, sample), and `Spectrum` (channel, frequency, time). `ScipyAudioEngine`
is the executable reference; `WandasAudioEngine` delays import so the unchanged
UI and SciPy route still load without wandas. Masks, selections and immutable
application undo/redo are separate pure modules. The legacy `c_map.py` remains
untouched, so current UI behavior is not switched by this PR.

SciPy uses normalized floating samples, a one-sided padded/boundary STFT, direct
ISTFT and explicit trim/pad to source length. This intentionally differs from the
legacy peak normalization (which produces NaN for silence), noverlap naming, and
iterative reconstruction. Direct ISTFT is the intended normal edit route because
the complex phase is retained. Real wandas measurements (length, RMS, peak,
maximum absolute error, spectral error and finite-value check) remain gated on a
verified pinned build; optional listening WAVs should be emitted only when those
metrics reveal a meaningful difference.

## Staged follow-up plan

1. With network access, inspect wandas source/tests, select a release (or record
   the exact main commit SHA and rationale), add a hash-locked native dependency,
   and separately pin the same browser wheel URLs in `pyscript.toml`.
2. Run the real wandas comparison matrix for mono/stereo, silence, short and tiny
   signals. Record tolerances from observed window/centering/scaling conventions.
3. Route the existing UI through `ScipyAudioEngine` without numerical/UI changes.
4. Switch the default to wandas only after native and Pyodide tests pass; keep the
   SciPy engine selectable during rollout.
5. Introduce an immutable editor state/view model and application `EditHistory`;
   commit only Select/Clear/Apply/Enhance actions and discard redo after commits.
6. Replace inheritance and polling with BrowserApp composition and release/commit/
   play/save reconstruction events, then remove duplicate reconstruction caches.
7. In later PRs remove the legacy Griffin–Lim and SciPy paths, and migrate the
   large historical requirements file to small locked native/browser groups.

## Risks

The concrete wandas array orientation, metadata property names, `from_numpy`
signature, `to_wav` return type, window naming, boundary padding, normalization,
and Pyodide wheel availability require verification against the pinned revision.
The adapter fails clearly rather than use private attributes. The existing UI
still has silence normalization and timer issues because changing it was expressly
out of scope. The present comparison test proves adapter orchestration against a
public-API test double, not numerical equivalence with the unavailable library.
