#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# twdtw is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Test cycle conversion utilities."""

import numpy as np

from twdtw import cycles_as_number


def test_day_of_year():
	"""Test day-of-year conversion."""
	# create dates
	dates = np.array(["2020-01-01", "2020-12-31", "2021-01-01"], dtype="datetime64[D]")

	# convert to numeric cycle
	numeric, length = cycles_as_number(dates, "year", "day")

	# check the length
	assert length == 366.0

	# check the numeric values
	np.testing.assert_array_equal(numeric, [1.0, 366.0, 1.0])  # 2020 is a leap year


def test_numeric_cycle_wraps_around_origin():
	"""Test numeric cycle wrapping around origin."""
	# create origin and dates
	origin = np.datetime64("2020-01-01")
	dates = origin + np.array([0, 10, 370], dtype="timedelta64[D]")

	# convert to numeric cycle
	numeric, length = cycles_as_number(
		dates, cycle_length=365, time_scale="day", origin=origin
	)

	# check the length
	assert length == 365.0

	# check the numeric values
	np.testing.assert_allclose(numeric, [0.0, 10.0, 5.0])


def test_numeric_passthrough():
	"""Test numeric cycle passthrough."""
	# convert to numeric cycle
	numeric, length = cycles_as_number([1.0, 100.0, 200.0], cycle_length=366)

	# check the numeric values
	np.testing.assert_array_equal(numeric, [1.0, 100.0, 200.0])

	# check the length
	assert length == 366.0
