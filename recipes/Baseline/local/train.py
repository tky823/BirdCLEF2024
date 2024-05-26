import audyn
from omegaconf import DictConfig

from birdclef2024.utils import setup_config
from birdclef2024.utils.driver import BaseTrainer


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    trainer = BaseTrainer.build_from_config(config)
    trainer.run()


if __name__ == "__main__":
    main()
