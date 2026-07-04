# %%
#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# twdtw is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

# %% [markdown]
# # Matching a pattern with plain numpy arrays
#
# The array form of `twdtw` takes a series and a pattern as arrays, each with its
# own observation times (`x_time` / `y_time`). This example builds a synthetic
# four-band season and matches a time-shifted copy of it.
#
# Run it with `uv run python examples/ts-numpy.py`, or open it as a notebook
# (Jupyter and VS Code read the `# %%` cells directly).

# %%
import numpy as np

from twdtw import logistic_weight, twdtw

# %% [markdown]
# ## 1) Build a one-season, four-band signal
#
# Twenty-three observations every 15 days (the MOD13Q1 cadence).

# %%
# set the random seed
np.random.seed(42)

# dates
dates = np.datetime64("2020-09-01") + np.arange(23) * np.timedelta64(15, "D")

# phase
phase = np.linspace(0, np.pi, 23)

# series
series = np.stack([np.sin(phase) * k for k in (2, 3, 4, 5)], axis=1)

# %% [markdown]
# ## 2) Use the same shape, shifted 30 days, as the pattern

# %%
# pattern
pattern = np.stack([np.sin(phase) * k for k in (2, 3, 4, 5)], axis=1)

# pattern dates
pattern_dates = dates + np.timedelta64(30, "D")

# %% [markdown]
# ## 3) Define the logistic time weight
#
# The `logistic_weight` function defined the `(steepness, midpoint)` in cycle
# days. This is the penalty added as two observations drift apart within the
# cycle.

# %%
weight = logistic_weight(0.1, 50)

# %% [markdown]
# ## 4) Dissimilarity of the best match over a yearly cycle

# %%
# compute the distance
distance = twdtw(
	x=series,
	y=pattern,
	x_time=dates,
	y_time=pattern_dates,
	time_weight=weight,
	cycle_length="year",
	time_scale="day",
)

# print the distance
print(f"best-match distance: {distance:.4f}")

# %% [markdown]
# ## 5) Every matching subinterval
#

# %%
# compute the matches
matches = twdtw(
	series,
	pattern,
	x_time=dates,
	y_time=pattern_dates,
	time_weight=weight,
	cycle_length="year",
	time_scale="day",
	output="matches",
)

# prepare matches for printing
rows = zip(
	matches.start,
	matches.end,
	matches.distance,
	strict=True,
)

# print the matches
for start, end, dist in rows:
	print(f"Match: columns {start}..{end}  distance {dist:.4f}")
