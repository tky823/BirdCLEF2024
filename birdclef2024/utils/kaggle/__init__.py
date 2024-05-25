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
    if UserSecretsClient is not None:
        user_secrets = UserSecretsClient()
        token = user_secrets.get_secret("HUGGINGFACE_TOKEN")
    else:
        token = os.getenv("HUGGINGFACE_TOKEN")

    return token


def load_huggingface_repo_id() -> Optional[str]:
    if UserSecretsClient is not None:
        user_secrets = UserSecretsClient()
        repo_id = user_secrets.get_secret("HUGGINGFACE_REPO_ID")
    else:
        repo_id = os.getenv("HUGGINGFACE_REPO_ID")

    return repo_id
