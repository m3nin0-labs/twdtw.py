## twdtw

Time-Weighted Dynamic Time Warping (TWDTW, [Maus et al. 2016](https://doi.org/10.1109/JSTARS.2016.2517118)) for Python. 

## Install

The package is managed with [uv](https://docs.astral.sh/uv/). A base installation builds the C++ core and pulls numpy:

```bash
uv sync
```

The polars frontend is optional. Enable it with the `polars` extra:

```bash
uv sync --extra polars
```

## Your first match (numpy)

Start from two aligned things: the observation times and the band values. Here is one season of a four-band signal sampled every 15 days.

```python
import numpy as np
from twdtw import twdtw, logistic_weight

# define dates
dates = np.datetime64("2020-09-01") + np.arange(23) * np.timedelta64(15, "D")

# define series
phase = np.linspace(0, np.pi, 23)
series  = np.stack([np.sin(phase) * k for k in (2, 3, 4, 5)], axis=1)

# define patterns
pattern = series.copy()
```

The time weight is a logistic `(steepness, midpoint)` pair. This is the penalty applied as two observations drift apart in the cycle.

```python
weight = logistic_weight(0.1, 50)
```

Now match the `pattern` against the `series`. Each series carries its own time via `x_time` / `y_time`, and `cycle_length="year"` wraps the dates onto a yearly cycle:

```python
twdtw(
  x = series,
  y = pattern,
  x_time=dates,
  y_time=dates,
  time_weight=weight,
  cycle_length="year"
)
```

To get every matching subinterval instead of just the best distance you can use the `output` parameter:

```python
twdtw(
  x = series,
  y = pattern,
  x_time=dates,
  y_time=dates,
  time_weight=weight,
  cycle_length="year",
  output="matches"
)
```

Note: Any array-like (lists, tuples, `numpy.ndarray`) works for the values (`x` and `y`) and times (`x_time` and `y_time`).

## From a DataFrame (polars)

With the `polars` installed, the same `twdtw` accepts polars frames directly. Time and band values already travel together in a frame, so there are no parallel time arrays to pass:

```python
import polars as pl

# define series
series = pl.DataFrame({"Index": dates, "NDVI": series[:, 0], "EVI": series[:, 1]})

# define pattern
pattern = series.clone()

twdtw(
  x = series,
  y = pattern,
  time="Index",
  bands=["NDVI", "EVI"],
  time_weight=weight,
  cycle_length="year"
)
```

`time` names the date column and `bands` selects the value columns (defaulting to every column but `time`). The result is identical to the array form.

## Learn more

Runnable [jupytext](https://jupytext.readthedocs.io) scripts live in `examples/` with usage examples:

- `examples/ts-numpy.py`, shows the numpy walkthrough presented above.
- `examples/ts-polars.py`, classifies a LULC time-series dataset from Cerrado using TWDTW and GAM.

## License & attribution

Code is MIT licensed.

Data used in the examples were extracted from the [sitsdata]() repository and are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
