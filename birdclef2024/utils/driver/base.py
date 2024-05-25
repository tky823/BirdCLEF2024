import os

from audyn.utils.driver import BaseTrainer as _BaseTrainer
from huggingface_hub import HfApi

from ..kaggle import load_huggingface_repo_id, load_huggingface_token

_token = load_huggingface_token()
_repo_id = load_huggingface_repo_id(load_huggingface_repo_id)


class BaseTrainer(_BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.huggingface_api = HfApi(token=_token)

    def save_checkpoint(self, save_path: str) -> None:
        super().save_checkpoint(save_path)

        self.upload_checkpoint(save_path)
        self.upload_tensorboard(self.writer.log_dir)

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
