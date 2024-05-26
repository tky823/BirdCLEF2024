import audyn
from omegaconf import DictConfig

from birdclef2024.utils import setup_config
from birdclef2024.utils.driver import BaseGenerator


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    generator = BaseGenerator.build_from_config(config)
    generator.run()


if __name__ == "__main__":
    main()
