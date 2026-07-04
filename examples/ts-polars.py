# %%
#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# twdtw is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

# %% [markdown]
# # Time-Weighted DTW with GAM seasonal patterns
#
# This example classifies Cerrado land-cover series with TWDTW, using a
# per-label Generalized Additive Model (GAM) as the reference pattern.
#
# The dataset (`examples/data/samples_cerrado_mod13q1.parquet`) holds MOD13Q1
# samples of the Brazilian Cerrado: for every point, a one-year NDVI series
# sampled every 16 days, tagged with a land-cover label. Each label owns many
# series, so we fit one GAM per label over the day-of-year to get a smooth
# seasonal NDVI pattern, and then score series against every pattern.
#
# Run it with `uv run python examples/ts-polars.py`, or open it as a notebook
# (Jupyter and VS Code read the `# %%` cells directly).

# %%
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from pygam import LinearGAM, s
from tqdm import tqdm

from twdtw import logistic_weight, twdtw

# %% [markdown]
# ## Configuration

# %%
# data directory
DATA = Path("data") / "samples_cerrado_mod13q1.parquet"

# band to model and match on
BAND = "NDVI"

# number of series pooled per label when fitting its GAM
FIT_SAMPLES = 200

# logistic time-weight parameters: (steepness, midpoint) in cycle days
WEIGHT = logistic_weight(0.1, 50)


# %% [markdown]
# ## Fitting a seasonal pattern per label
#
# `patterns()` pools a sample of each label's series, fits a GAM of the band
# value over the **day-of-year**, and predicts it on a regular yearly grid. The
# result is one smooth seasonal curve per label, stacked into a long table.


# %%
def patterns(
	frame: pl.DataFrame,
	band: str = BAND,
	n_points: int = 23,
	fit_samples: int = FIT_SAMPLES,
	seed: int = 0,
) -> pl.DataFrame:
	"""Fit one GAM seasonal pattern per label.

	For each label the series are pooled and a GAM is fit, then sampled on a
	regular yearly grid to give a smooth seasonal curve.

	Args:
		frame (pl.DataFrame): Long sample table with ``label``, ``sample_id``,
			an ``Index`` date column and the ``band`` value column.

		band (str): Value column to model (e.g. ``"NDVI"``).

		n_points (int): Number of points to sample each pattern at across a year.

		fit_samples (int): Number of series pooled per label before fitting.

		seed (int): Seed for the sampling of the fitting pool.

	Returns:
		pl.DataFrame: A long patterns table with columns ``label``, ``Index``
			(a canonical year grid) and ``band``, holding one smoothed curve per
			label.
	"""
	# set the random seed
	rng = np.random.default_rng(seed)

	# a regular day-of-year grid and the canonical dates
	grid = np.linspace(1, 366, n_points)

	# convert the grid to dates
	dates = [date(2019, 1, 1) + timedelta(days=int(day) - 1) for day in grid]

	curves = []
	for label in frame["label"].unique(maintain_order=True):
		# get unique sample IDs for the label
		ids = frame.filter(pl.col("label") == label)["sample_id"].unique().to_numpy()

		# sample series
		pool = rng.choice(ids, size=min(fit_samples, len(ids)), replace=False)

		# get the selected series
		pooled = frame.filter(pl.col("sample_id").is_in(pool))

		# convert the dates to ordinal days
		day_of_year = pooled["Index"].dt.ordinal_day().to_numpy().reshape(-1, 1)

		# fit the GAM
		gam = LinearGAM(s(0)).fit(day_of_year, pooled[band].to_numpy())

		# predict on the grid
		curve = gam.predict(grid)

		# save the curve
		curves.append(pl.DataFrame({"label": label, "Index": dates, band: curve}))

	# return curves
	return pl.concat(curves)


# %% [markdown]
# ## 1) Load the Cerrado NDVI samples

# %%
frame = pl.read_parquet(DATA).select("sample_id", "label", "Index", BAND)

frame.head()

# %% [markdown]
# ## 2) Fit one GAM seasonal pattern per label
#
# Each label becomes one smooth NDVI curve over a canonical year.
#

# %%
# generate GAM
pattern_table = patterns(frame)

# splits the long table into one frame per label
pattern_frames = pattern_table.partition_by("label", maintain_order=True)

# show table
pattern_table

# %% [markdown]
# ## 3) Split the dataset into individual series
#
# We score the *entire* sample set: one date-sorted frame per series, each still
# carrying its land-cover label.

# %%
series_list = frame.sort("sample_id", "Index").partition_by(
	"sample_id", maintain_order=True
)

print(f"series to score: {len(series_list)}")

# %% [markdown]
# ## 4) TWDTW distance of series to label pattern
#
# Each row is a series and each label column is the TWDTW dissimilarity to a
# label pattern. The result is a `nrow x npatterns` distance table. The
# smallest value in a row is that series' nearest seasonal pattern.

# %%
records = []

# for each series
for series in tqdm(series_list, desc="scoring series"):
	# initialize the row
	row = dict(
		sample_id=int(series["sample_id"][0]),
		label=series["label"][0],
	)

	# for each pattern
	for pattern in pattern_frames:
		# compute the distance
		distance = twdtw(
			x=series,
			y=pattern,
			time_weight=WEIGHT,
			cycle_length="year",
			bands=[BAND],
		)

		# save the distance
		row[pattern["label"][0]] = round(distance, 2)

	# save the row
	records.append(row)

# convert the records to a DataFrame
distance_table = pl.DataFrame(records)

# show the distance table
distance_table

# %% [markdown]
# ## 5) Nearest-pattern accuracy
#
# The predicted class is the pattern with the smallest distance. Comparing it to
# the true label gives a quick baseline for this single-band (NDVI) setup.

# %%
# get the label columns
label_cols = [pattern["label"][0] for pattern in pattern_frames]

# unpivot the distance table
predicted = (
	distance_table.unpivot(
		index=["sample_id", "label"],
		on=label_cols,
		variable_name="pattern",
		value_name="distance",
	)
	.group_by("sample_id", "label", maintain_order=True)
	.agg(pl.col("pattern").sort_by("distance").first().alias("predicted"))
)

# compute the accuracy
accuracy = (predicted["label"] == predicted["predicted"]).mean()

# print the accuracy
print(f"nearest-pattern accuracy: {accuracy:.1%}")
