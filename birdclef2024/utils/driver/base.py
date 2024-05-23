import os

from audyn.utils.driver import BaseTrainer as _BaseTrainer
from huggingface_hub import HfApi

_token = os.getenv("HUGGINGFACE_TOKEN")
_repo_id = os.getenv("HUGGINGFACE_REPO_ID")


class BaseTrainer(_BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.huggingface_api = HfApi(token=_token)

    def save_checkpoint(self, save_path: str) -> None:
        super().save_checkpoint(save_path)

        self.upload_checkpoint(save_path)

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
