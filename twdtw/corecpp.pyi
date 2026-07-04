#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# twdtw is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Type stub for the nanobind ``corecpp`` extension."""

import numpy as np

def twdtw_core(
	XM: np.ndarray,
	YM: np.ndarray,
	CM: np.ndarray,
	DM: np.ndarray,
	VM: np.ndarray,
	JB: np.ndarray,
	alpha: float,
	beta: float,
	max_elapsed: float,
	cycle_length: float,
) -> None:
	"""Run the TWDTW dynamic algorithm."""
