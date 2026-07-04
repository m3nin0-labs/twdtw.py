#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# twdtw is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Test time-weighting utilities."""

from twdtw import logistic_weight


def test_logistic_weight_returns_float_pair():
	"""Test logistic weight returns float pair."""
	# arrange / act
	steepness, midpoint = logistic_weight(0.1, 50)

	# assert
	assert (steepness, midpoint) == (0.1, 50.0)
	assert isinstance(steepness, float)
	assert isinstance(midpoint, float)
