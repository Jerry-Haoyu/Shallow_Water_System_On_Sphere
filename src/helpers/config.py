import sys

import yaml


def load_raw_config(section, defaults, config_path=None):
    """Merge the ``section`` block of a YAML config file over ``defaults``.

    ``config_path`` defaults to ``sys.argv[1]`` (or "config.yml" if no CLI
    argument was given), matching the run_solver.py / inference.py convention.
    Returns a plain dict (not yet a SimpleNamespace) so callers can still
    normalize/coerce individual fields before constructing one.
    """
    if config_path is None:
        config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    return defaults | (config.get(section) or {})
