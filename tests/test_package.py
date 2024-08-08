from __future__ import annotations

import importlib.metadata

import openlifu_registration as m


def test_version():
    assert importlib.metadata.version("openlifu_registration") == m.__version__
