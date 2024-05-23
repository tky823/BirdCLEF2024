import os

from audyn.utils.driver import BaseTrainer as _BaseTrainer
from huggingface_hub import upload_file

_token = os.getenv("HUGGINGFACE_TOKEN")
_repo_id = os.getenv("HUGGINGFACE_REPO_ID")


class BaseTrainer(_BaseTrainer):
    def save_checkpoint(self, save_path: str) -> None:
        super().save_checkpoint(save_path)

        self.upload_checkpoint(save_path)

    def upload_checkpoint(self, path: str) -> None:
        cwd = os.getcwd()
        recipe_name = os.path.basename(cwd)
        path_in_repo = os.path.join("recipes", recipe_name, path)
        repo_type = "model"

        try:
            upload_file(
                path_or_fileobj=path,
                path_in_repo=path_in_repo,
                repo_id=_repo_id,
                token=_token,
                repo_type=repo_type,
            )
        except Exception:
            # give up uploading
            self.logger.info(f"Failed in uploading {path_in_repo}")
