#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# twdtw is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Cycle conversion utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#
# Conversion constants for each time scale.
#
SECONDS = {
	"second": 1.0,
	"minute": 60.0,
	"hour": 3600.0,
	"day": 86400.0,
	"week": 604800.0,
}


def _named_cycle(
	time: np.ndarray, cycle_length: str, time_scale: str
) -> tuple[np.ndarray, float]:
	"""Resolve a calendar cycle such as day-of-year.

	Args:
		time (np.ndarray): Observation times as ``datetime64[s]``.

		cycle_length (str): Calendar cycle name (currently only ``"year"``).

		time_scale (str): Unit of the cycle (currently only ``"day"``).

	Returns:
		tuple[np.ndarray, float]: The in-cycle numeric positions and the cycle
			length.

	Raises:
		ValueError: If the ``cycle_length`` / ``time_scale`` pair is unsupported.
	"""
	# handle year / day conversion
	if cycle_length == "year" and time_scale == "day":
		# convert year to days
		year = time.astype("datetime64[Y]").astype("datetime64[D]")

		# subtract year from time to get day of year
		day_of_year = (time.astype("datetime64[D]") - year) / np.timedelta64(1, "D")

		# return day cycle length
		return (day_of_year + 1.0).astype(np.float64), 366.0

	# fallback: unsupported cycle length
	raise ValueError(
		f"unsupported cycle_length={cycle_length!r} with time_scale={time_scale!r}"
	)


def cycles_as_number(
	time: npt.ArrayLike,
	cycle_length: str | float,
	time_scale: str = "day",
	origin: np.datetime64 | str | None = None,
) -> tuple[np.ndarray, float]:
	"""Convert observation times to numeric positions within a temporal cycle.

	Args:
		time (npt.ArrayLike): Observation times, numeric or date-like.

		cycle_length (str | float): Calendar cycle name or a numeric cycle
			length.

		time_scale (str): Unit used when wrapping a numeric ``cycle_length``.

		origin (np.datetime64 | str | None): Reference date for numeric cycles.
			`None` uses the earliest time.

	Returns:
		tuple[np.ndarray, float]: The in-cycle numeric positions and the
			resolved numeric cycle length.
	"""
	# convert to numpy array
	time = np.asarray(time)

	# if numeric, return as is
	if np.issubdtype(time.dtype, np.number):
		return time.astype(np.float64), float(cycle_length)

	# convert to datetime64 (seconds)
	time = time.astype("datetime64[s]")

	# if cycle length is a string, use the named cycle
	if isinstance(cycle_length, str):
		return _named_cycle(time, cycle_length, time_scale)

	# if origin is not provided, use the earliest time
	if origin is None:
		origin = time.min()

	# convert origin to datetime64 (seconds)
	origin = np.datetime64(origin).astype("datetime64[s]")

	# calculate elapsed time
	# logic: convert to seconds and divide by the time scale
	elapsed = (time - origin) / np.timedelta64(1, "s") / SECONDS[time_scale]

	# return the modulo of the elapsed time and the cycle length
	return np.mod(elapsed, cycle_length).astype(np.float64), float(cycle_length)
