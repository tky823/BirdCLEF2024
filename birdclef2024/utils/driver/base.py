import os

import torch
from audyn.utils.driver import BaseGenerator as _BaseGenerator
from audyn.utils.driver import BaseTrainer as _BaseTrainer
from huggingface_hub import HfApi
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm

from ..kaggle import load_huggingface_repo_id, load_huggingface_token

_token = load_huggingface_token()
_repo_id = load_huggingface_repo_id()


class BaseTrainer(_BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.huggingface_api = HfApi(token=_token)

    def save_checkpoint(self, save_path: str) -> None:
        super().save_checkpoint(save_path)

        self.upload_checkpoint(save_path)
        self.upload_tensorboard(self.writer.log_dir)
        self.upload_log(HydraConfig.get().run.dir)

    def upload_checkpoint(self, path: str) -> None:
        cwd = os.getcwd()
        recipe_name = os.path.basename(cwd)
        path_in_repo = os.path.join("recipes", recipe_name, path)
        repo_type = "model"

        try:
            self.huggingface_api.upload_file(
                path_or_fileobj=path,
                path_in_repo=path_in_repo,
                repo_id=_repo_id,
                repo_type=repo_type,
                run_as_future=True,
            )
        except Exception:
            # give up uploading
            self.logger.info(f"Failed in uploading {path_in_repo}")

    def upload_tensorboard(self, log_dir: str) -> None:
        cwd = os.getcwd()
        recipe_name = os.path.basename(cwd)
        path_in_repo = os.path.join("recipes", recipe_name, log_dir)
        repo_type = "model"

        try:
            self.huggingface_api.upload_folder(
                folder_path=log_dir,
                path_in_repo=path_in_repo,
                repo_id=_repo_id,
                repo_type=repo_type,
                run_as_future=True,
            )
        except Exception:
            # give up uploading
            self.logger.info(f"Failed in uploading {path_in_repo}")

    def upload_log(self, log_dir: str) -> None:
        cwd = os.getcwd()
        recipe_name = os.path.basename(cwd)
        path_in_repo = os.path.join("recipes", recipe_name, log_dir)
        repo_type = "model"

        try:
            self.huggingface_api.upload_folder(
                folder_path=log_dir,
                path_in_repo=path_in_repo,
                repo_id=_repo_id,
                repo_type=repo_type,
                run_as_future=True,
            )
        except Exception:
            # give up uploading
            self.logger.info(f"Failed in uploading {path_in_repo}")


class BaseGenerator(_BaseGenerator):
    @torch.no_grad()
    def run(self) -> None:
        test_config = self.config.test
        key_mapping = test_config.key_mapping.inference

        self.model.eval()

        for named_data in tqdm(self.loader):
            named_data = self.move_data_to_device(named_data, self.device)
            named_input = self.map_to_named_input(named_data, key_mapping=key_mapping)
            named_identifier = self.map_to_named_identifier(named_data, key_mapping=key_mapping)

            if hasattr(self.unwrapped_model, "inference"):
                output = self.unwrapped_model.inference(**named_input)
            else:
                output = self.unwrapped_model(**named_input)

            named_output = self.map_to_named_output(output, key_mapping=key_mapping)

            self.save_inference_torch_dump_if_necessary(
                named_output,
                named_data,
                named_identifier,
                config=test_config.output,
            )
            self.save_inference_audio_if_necessary(
                named_output,
                named_data,
                named_identifier,
                config=test_config.output,
            )
            self.save_inference_spectrogram_if_necessary(
                named_output,
                named_data,
                named_identifier,
                config=test_config.output,
            )
