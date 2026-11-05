"""Ustawienia projektu Kedro (Sprint 2).

Konfiguracja leży w ``conf/``:
- ``kedro run``           -> ``base`` + ``local`` (pełne dane),
- ``kedro run --env ci``  -> ``base`` + ``ci`` (próbka ``data/01_raw/sample.csv``).
"""

from kedro.config import OmegaConfigLoader

CONF_SOURCE = "conf"
CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
}
