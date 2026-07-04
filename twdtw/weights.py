#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# twdtw is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Time-weighting utilities for TWDTW."""

from __future__ import annotations


def logistic_weight(steepness: float, midpoint: float) -> tuple[float, float]:
	"""Logistic time-weight parameters.

	The penalty added to the band distance is
	`1 / (1 + exp(-alpha * (dt - beta)))`, where `dt` is the cyclic elapsed
	time between two observations.

	Args:
		steepnes (float): Logistic steepness `alpha`.

		midpoint (float): Logistic midpoint `beta`.

	Returns:
		The `(steepness, midpoint)` pair as floats.
	"""
	return float(steepness), float(midpoint)
