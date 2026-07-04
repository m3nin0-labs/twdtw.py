#
# Copyright (C) 2026 Felipe Carlos (m3nin0-labs).
#
# twdtw is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

import importlib

#
# Frontend modules.
#
FRONTENDS = ("twdtw.frontends.polr",)


def load_frontends() -> None:
	"""Import frontend modules."""
	for module in FRONTENDS:
		try:
			importlib.import_module(module)
		except ImportError:
			continue
