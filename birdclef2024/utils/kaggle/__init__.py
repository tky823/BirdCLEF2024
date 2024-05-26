import os
from typing import Optional

try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None

__all__ = [
    "load_huggingface_token",
    "load_huggingface_repo_id",
]


def load_huggingface_token() -> Optional[str]:
    token = None

    if UserSecretsClient is not None:
        user_secrets = UserSecretsClient()

        try:
            token = user_secrets.get_secret("HUGGINGFACE_TOKEN")
        except Exception:
            # if HUGGINGFACE_TOKEN is not defined
            pass

    if token is None:
        token = os.getenv("HUGGINGFACE_TOKEN")

    return token


def load_huggingface_repo_id() -> Optional[str]:
    repo_id = None

    if UserSecretsClient is not None:
        user_secrets = UserSecretsClient()

        try:
            repo_id = user_secrets.get_secret("HUGGINGFACE_REPO_ID")
        except Exception:
            # if HUGGINGFACE_REPO_ID is not defined
            pass

    if repo_id is None:
        repo_id = os.getenv("HUGGINGFACE_REPO_ID")

    return repo_id
