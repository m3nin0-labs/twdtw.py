#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# twdtw is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Test TWDTW implementation."""

import numpy as np
import pytest

from twdtw import Matches, logistic_weight, twdtw

WEIGHT = logistic_weight(0.1, 50)


def make_season(rng, shift_days=0, noise=0.1):
	"""A one-season, four-band signal sampled every 15 days."""
	# create dates
	dates = np.datetime64("2020-09-01") + np.arange(23) * np.timedelta64(15, "D")

	# shift dates
	dates = dates + np.timedelta64(shift_days, "D")

	# create phase
	phase = np.linspace(0, np.pi, 23)

	# create bands
	bands = np.stack([np.sin(phase) * k for k in (2, 3, 4, 5)], axis=1)

	# add noise
	return dates, bands + rng.normal(scale=noise, size=bands.shape)


def test_matching_pattern_beats_mismatch():
	"""Test matching pattern beats mismatch."""
	# create random number generator
	rng = np.random.default_rng(0)

	tx, x = make_season(rng)
	ty, y = make_season(rng, shift_days=30)  # same shape, shifted in time
	_, noise = make_season(rng, noise=3.0)

	# compute the distances
	good = twdtw(
		x=x,
		y=y,
		x_time=tx,
		y_time=ty,
		time_weight=WEIGHT,
		cycle_length="year",
	)

	bad = twdtw(
		x,
		noise,
		x_time=tx,
		y_time=ty,
		time_weight=WEIGHT,
		cycle_length="year",
	)

	# check the distances
	assert good < bad


def test_distance_is_min_of_matches():
	"""Test distance is min of matches."""
	# create random number generator
	rng = np.random.default_rng(1)
	tx, x = make_season(rng)
	ty, y = make_season(rng, shift_days=30)

	# compute the matches
	matches = twdtw(
		x=x,
		y=y,
		x_time=tx,
		y_time=ty,
		time_weight=WEIGHT,
		cycle_length="year",
		output="matches",
	)

	# compute the distance
	distance = twdtw(
		x=x,
		y=y,
		x_time=tx,
		y_time=ty,
		time_weight=WEIGHT,
		cycle_length="year",
	)

	# check the matches
	assert isinstance(matches, Matches)

	# check the matches shape
	assert matches.start.shape == matches.end.shape == matches.distance.shape

	# check the distance
	assert distance == pytest.approx(matches.distance.min())


def test_internals_expose_matrices():
	"""Test internals expose matrices."""
	# create random number generator
	rng = np.random.default_rng(2)

	tx, x = make_season(rng)
	ty, y = make_season(rng, shift_days=30)

	# compute the internals
	internals = twdtw(
		x=x,
		y=y,
		x_time=tx,
		y_time=ty,
		time_weight=WEIGHT,
		cycle_length="year",
		output="internals",
	)

	# check the matrices shape
	assert internals.CM.shape == (y.shape[0] + 1, x.shape[0])

	# check the cycle length
	assert internals.cycle_length == 366.0


def test_polars_frame_matches_array():
	"""Test polars frame matches array."""
	# create random number generator
	pl = pytest.importorskip("polars")

	rng = np.random.default_rng(3)
	tx, x = make_season(rng)
	ty, y = make_season(rng, shift_days=30)

	# create bands
	bands = ["NDVI", "EVI", "NIR", "MIR"]

	# compute the matches from array
	from_array = twdtw(
		x, y, x_time=tx, y_time=ty, time_weight=WEIGHT, cycle_length="year"
	)

	# create polars frames
	series = pl.DataFrame({"Index": tx, **{b: x[:, i] for i, b in enumerate(bands)}})
	pattern = pl.DataFrame({"Index": ty, **{b: y[:, i] for i, b in enumerate(bands)}})

	# compute the matches from frame
	from_frame = twdtw(series, pattern, time_weight=WEIGHT, cycle_length="year")

	# check the matches
	assert from_frame == pytest.approx(from_array)
