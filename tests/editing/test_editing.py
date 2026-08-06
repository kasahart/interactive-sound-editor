import numpy as np
from interactive_sound_editor.editing.history import EditHistory
from interactive_sound_editor.editing.mask import (db_attenuation, frequency_trend_mask, harmonic_mask, percussive_mask, split_masks, zscore_mask)
from interactive_sound_editor.editing.selection import TimeFrequencyRegion, region_to_slices


def test_mask_operations_are_finite_for_silence():
    values = np.zeros((1, 9, 7), complex)
    for result in (harmonic_mask(values, 3), percussive_mask(values, 3), zscore_mask(values), frequency_trend_mask(values)):
        assert result.shape == values.shape
        assert np.isfinite(result).all()


def test_db_mask_and_selected_residual_relation():
    mask = np.array([0, 1, np.nan])
    selected, residual = split_masks(mask)
    np.testing.assert_allclose(selected + residual, 1)
    np.testing.assert_allclose(db_attenuation(np.array([False, True]), 20), [1, .1])


def test_none_empty_and_valid_selection():
    times = np.array([0., .1, .2]); frequencies = np.array([0., 100., 200.])
    assert region_to_slices(None, times, frequencies) is None
    empty = region_to_slices(TimeFrequencyRegion(1, 2, 1000, 2000), times, frequencies)
    assert empty == (slice(0, 0), slice(0, 0))
    assert region_to_slices(TimeFrequencyRegion(.2, 0, 200, 0), times, frequencies) == (slice(0, 3), slice(0, 3))


def test_undo_redo_and_new_commit_discards_future():
    history = EditHistory((), "a").commit("b").commit("c")
    assert history.undo().current == "b"
    undone = history.undo()
    assert undone.redo().current == "c"
    assert undone.commit("d").future == ()
