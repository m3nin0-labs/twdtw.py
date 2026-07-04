#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# twdtw is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Optional polars frontend."""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

from twdtw.handler import Internals, Matches, twdtw_dispatch


@twdtw_dispatch.register(pl.DataFrame)
def _twdtw_frame(
	x: pl.DataFrame,
	y: pl.DataFrame,
	time_weight: tuple[float, float],
	cycle_length: str | float,
	time: str = "Index",
	bands: list[str] | None = None,
	time_scale: str = "day",
	origin: np.datetime64 | str | None = None,
	max_elapsed: float = np.inf,
	output: Literal["distance", "matches", "internals"] = "distance",
) -> float | Matches | Internals:
	"""Execute TWDTW on two polars frames.

	Args:
		x (pl.DataFrame): The long time series, one row per observation.

		y (pl.DataFrame): The temporal pattern to match.

		time_weight (tuple[float, float]): Logistic ``(steepness, midpoint)``
										   time-weight parameters.

		cycle_length (str | float): Temporal cycle as a name (e.g. ``"year"``)
									or a number.

		time (str): Name of the time column in both frames.

		bands (list[str] | None): Band columns to match on. ``None`` uses
								   every column except ``time``.

		time_scale (str): Unit used when wrapping a numeric ``cycle_length``.

		origin (np.datetime64 | str | None): Reference date for numeric
											 cycles. ``None`` uses the earliest
											 time.

		max_elapsed (float): Maximum cyclic elapsed time allowed for a local
			match.

		output (Literal["distance", "matches", "internals"]): One of
			``"distance"``, ``"matches"`` or ``"internals"``.

	Returns:
		float | Matches | Internals: Same result as the array form of `twdtw.twdtw`.
	"""
	# resolve the band columns
	columns = bands if bands is not None else [c for c in x.columns if c != time]

	# extract the times
	x_time = x[time].to_numpy()
	y_time = y[time].to_numpy()

	# execute TWDTW
	return twdtw_dispatch(
		x.select(columns).to_numpy(),
		y.select(columns).to_numpy(),
		x_time=x_time,
		y_time=y_time,
		time_weight=time_weight,
		cycle_length=cycle_length,
		time_scale=time_scale,
		origin=origin,
		max_elapsed=max_elapsed,
		output=output,
	)
