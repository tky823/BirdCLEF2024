import warnings

from audyn.utils import setup_config as _setup_config
from omegaconf import DictConfig


def setup_system(config: DictConfig) -> None:
    """Set up config before training and evaluation.

    Args:
        config (DictConfig): Config to set up.

    .. note::

        This function is deprecated. Use ``birdclef2024.utils.setup_config()`` instead.

    """
    warnings.warn(
        "birdclef2024.utils.setup_system is deprecated. Use "
        "birdclef2024.utils.setup_config instead.",
        stacklevel=2,
    )
    setup_config(config)


def setup_config(config: DictConfig) -> None:
    """Set up config before training and evaluation.

    Args:
        config (DictConfig): Config to set up.

    """
    _setup_config(config)
