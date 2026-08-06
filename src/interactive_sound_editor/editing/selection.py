from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TimeFrequencyRegion:
    start_time: float
    end_time: float
    low_frequency: float
    high_frequency: float


def region_to_slices(region: TimeFrequencyRegion | None, times: np.ndarray, frequencies: np.ndarray) -> tuple[slice, slice] | None:
    if region is None:
        return None
    t0, t1 = sorted((region.start_time, region.end_time))
    f0, f1 = sorted((region.low_frequency, region.high_frequency))
    ti = np.flatnonzero((times >= t0) & (times <= t1))
    fi = np.flatnonzero((frequencies >= f0) & (frequencies <= f1))
    if ti.size == 0 or fi.size == 0:
        return slice(0, 0), slice(0, 0)
    return slice(fi[0], fi[-1] + 1), slice(ti[0], ti[-1] + 1)
