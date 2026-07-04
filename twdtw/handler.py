#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# twdtw is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""twdtw handler layer."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt

import twdtw.corecpp as corecpp
from twdtw.cycles import cycles_as_number


class Matches(NamedTuple):
	"""Best-matching subintervals of the series for one pattern.

	Attributes:
		start: Series column where each match begins.

		end: Series column where each match ends.

		distance: TWDTW dissimilarity of each match.
	"""

	start: np.ndarray
	"""Series column where each match begins."""

	end: np.ndarray
	"""Series column where each match ends."""

	distance: np.ndarray
	"""TWDTW dissimilarity of each match."""


@dataclass
class Internals:
	"""Full state of a TWDTW run, for inspection or path reconstruction.

	Attributes:
		XM (np.ndarray): Prepared series matrix.

		YM (np.ndarray): Prepared pattern matrix.

		CM (np.ndarray): Accumulated-cost matrix.

		DM (np.ndarray): Step-direction matrix.

		VM (np.ndarray): Path-origin matrix.

		JB (np.ndarray): End column of each best match.

		matches (Matches): The reduced `Matches`.

		time_weight (tuple[float, float]): Logistic ``(steepness, midpoint)``
			parameters.

		max_elapsed (float): Maximum cyclic elapsed time allowed for a local
			match.

		cycle_length (float): Resolved numeric cycle length.
	"""

	XM: np.ndarray
	"""Prepared series matrix."""

	YM: np.ndarray
	"""Prepared pattern matrix."""

	CM: np.ndarray
	"""Accumulated-cost matrix."""

	DM: np.ndarray
	"""Step-direction matrix."""

	VM: np.ndarray
	"""Path-origin matrix."""

	JB: np.ndarray
	"""End column of each best match."""

	matches: Matches
	"""The reduced `Matches`."""

	time_weight: tuple[float, float]
	"""Logistic ``(steepness, midpoint)`` parameters."""

	max_elapsed: float
	"""Maximum cyclic elapsed time allowed for a local match."""

	cycle_length: float
	"""Resolved numeric cycle length."""


def _prepare(
	values: npt.ArrayLike,
	time: npt.ArrayLike,
	cycle_length: str | float,
	time_scale: str,
	origin: np.datetime64 | str | None,
) -> tuple[np.ndarray, float]:
	"""Prepare the input data for the TWDTW algorithm.

	Args:
		values (npt.ArrayLike): Band values shaped ``(n_obs,)`` or
			                    ``(n_obs, n_bands)``.

		time (npt.ArrayLike): Observation times aligned with ``values``.

		cycle_length (str | float): Temporal cycle as a name (e.g. ``"year"``)
			                        or a number.

		time_scale (str): Unit used when wrapping a numeric ``cycle_length``.

		origin (np.datetime64 | str | None): Reference date for numeric cycles.
				                             `None` uses the earliest time.

	Returns:
		tuple[np.ndarray, float]: The matrix and the resolved numeric cycle length.
	"""
	# convert to numpy array
	values = np.asarray(values, dtype=np.float64)

	# if values is 1D, convert to 2D
	if values.ndim == 1:
		values = values[:, None]

	# convert time to numeric positions within the cycle
	numeric_time, length = cycles_as_number(time, cycle_length, time_scale, origin)

	# stack the numeric time and values
	matrix = np.column_stack([numeric_time, values])

	# drop rows containing any NaN
	matrix = matrix[~np.isnan(matrix).any(axis=1)]

	# return the contiguous float64 matrix and the
	# resolved numeric cycle length
	return np.ascontiguousarray(matrix), length


def _shape_output(
	output: str,
	CM: np.ndarray,
	DM: np.ndarray,
	VM: np.ndarray,
	JB: np.ndarray,
	XM: np.ndarray,
	YM: np.ndarray,
	N: int,
	time_weight: tuple[float, float],
	max_elapsed: float,
	length: float,
) -> float | Matches | Internals:
	"""Reduce the raw TWDTW matrices into the requested output form.

	Args:
		output (Literal["distance", "matches", "internals"]): One of
			``"distance"``, ``"matches"`` or ``"internals"``.

		CM (np.ndarray): Accumulated-cost matrix, shape ``(N + 1, M)``.

		DM (np.ndarray): Step-direction matrix, shape ``(N + 1, M)``.

		VM (np.ndarray): Path-origin matrix, shape ``(N + 1, M)``.

		JB (np.ndarray): End column of each best match (``-1`` for empty slots).

		XM (np.ndarray): Prepared series matrix.

		YM (np.ndarray): Prepared pattern matrix.

		N (int): Number of pattern observations.

		time_weight (tuple[float, float]): Logistic ``(steepness, midpoint)``
			parameters.

		max_elapsed (float): Maximum cyclic elapsed time allowed for a local
			match.

		length (float): Resolved numeric cycle length.

	Returns:
		float | Matches | Internals: The minimum distance, the `Matches`,
			or the `Internals`.

	Raises:
		ValueError: If `output` is not a recognised value.
	"""
	# get the end columns of the best matches
	cols = JB[JB >= 0]

	# create the Matches object
	matches = Matches(
		start=VM[N, cols].copy(),
		end=cols.copy(),
		distance=CM[N, cols].copy(),
	)

	# case: distance
	if output == "distance":
		return float(matches.distance.min()) if cols.size else float("inf")

	# case: matches
	if output == "matches":
		return matches

	# case: internals
	if output == "internals":
		return Internals(
			XM, YM, CM, DM, VM, JB, matches, time_weight, max_elapsed, length
		)

	# fallback: unknown output
	raise ValueError(f"unknown output {output!r}")


@singledispatch
def twdtw_dispatch(
	x: npt.ArrayLike,
	y: npt.ArrayLike,
	*,
	x_time: npt.ArrayLike,
	y_time: npt.ArrayLike,
	time_weight: tuple[float, float],
	cycle_length: str | float,
	time_scale: str = "day",
	origin: np.datetime64 | str | None = None,
	max_elapsed: float = np.inf,
	output: Literal["distance", "matches", "internals"] = "distance",
) -> float | Matches | Internals:
	"""Array TWDTW implementation and `functools.singledispatch` base.

	See `twdtw` for the full parameter documentation.
	"""
	# get the logistic parameters
	alpha, beta = time_weight

	# prepare the series and pattern matrices
	XM, length = _prepare(x, x_time, cycle_length, time_scale, origin)
	YM, _ = _prepare(y, y_time, cycle_length, time_scale, origin)

	# get the number of series and pattern observations
	M, N = XM.shape[0], YM.shape[0]

	# allocate the accumulated-cost, step-direction, path-origin, and
	# best-match matrices
	CM = np.empty((N + 1, M), dtype=np.float64)
	DM = np.empty((N + 1, M), dtype=np.int32)
	VM = np.empty((N + 1, M), dtype=np.int32)
	JB = np.empty(M, dtype=np.int32)  # at most M match groups

	# TWDTW algorithm
	corecpp.twdtw_core(
		XM=XM,
		YM=YM,
		CM=CM,
		DM=DM,
		VM=VM,
		JB=JB,
		alpha=float(alpha),
		beta=float(beta),
		max_elapsed=float(max_elapsed),
		cycle_length=float(length),
	)

	# shape the output
	return _shape_output(
		output=output,
		CM=CM,
		DM=DM,
		VM=VM,
		JB=JB,
		XM=XM,
		YM=YM,
		N=N,
		time_weight=(alpha, beta),
		max_elapsed=max_elapsed,
		length=length,
	)


def twdtw(*args: object, **kwargs: object) -> float | Matches | Internals:
	"""Time-Weighted Dynamic Time Warping (TWDTW).

	This function matches temporal pattern ``y`` against series ``x`` using
	TWDTW.

	Args:
		x (npt.ArrayLike): The long time series, one row per observation.

		y (npt.ArrayLike): The temporal pattern to match.

		x_time (npt.ArrayLike): Observation times for ``x``.

		y_time (npt.ArrayLike): Observation times for ``y``.

		time_weight (tuple[float, float]): Logistic ``(steepness, midpoint)``
										   time-weight parameters.

		cycle_length (str | float): Temporal cycle as a name (e.g. ``"year"``)
									or a number.

		time_scale (str): Unit used when wrapping a numeric ``cycle_length``.

		origin (np.datetime64 | str | None): Reference date for numeric cycles.
											 ``None`` uses the earliest time.

		max_elapsed (float): Maximum cyclic elapsed time allowed for a local
							 match.

		output (Literal["distance", "matches", "internals"]): One of
			``"distance"``, ``"matches"`` or ``"internals"``.

	Returns:
		float | Matches | Internals: The minimum match distance (``float``) for
			``"distance"``, a `Matches` for ``"matches"``, or an
			`Internals` for ``"internals"``.

	Raises:
		TypeError: If the series ``x`` is not provided.

		ValueError: If `output` is not a recognised value.
	"""
	# resolve the dispatch subject (the series ``x``) from either call form
	if args:
		subject = args[0]

	elif "x" in kwargs:
		subject = kwargs["x"]

	else:
		raise TypeError("twdtw() missing required argument: 'x'")

	# dispatch on the type of the series and forward every argument
	return twdtw_dispatch.dispatch(type(subject))(*args, **kwargs)


# expose the dispatch machinery so frontends can overload the dispatcher
twdtw.register = twdtw_dispatch.register  # type: ignore[attr-defined]
twdtw.registry = twdtw_dispatch.registry  # type: ignore[attr-defined]
