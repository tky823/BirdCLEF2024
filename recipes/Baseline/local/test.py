import audyn
from audyn.utils import setup_config
from audyn.utils.driver import BaseGenerator
from omegaconf import DictConfig


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    generator = BaseGenerator.build_from_config(config)
    generator.run()


if __name__ == "__main__":
    main()
