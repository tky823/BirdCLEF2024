import os
from typing import Optional

try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None

__all__ = [
    "load_huggingface_token",
    "load_huggingface_repo_id",
    "is_on_kaggle",
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

    if token is not None and len(token) == 0:
        token = None

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

    if repo_id is not None and len(repo_id) == 0:
        repo_id = None

    return repo_id


def requires_huggingface_token() -> bool:
    token = None

    if UserSecretsClient is not None:
        user_secrets = UserSecretsClient()

        try:
            token = user_secrets.get_secret("REQUIRES_HUGGINGFACE_TOKEN")
        except Exception:
            # if HUGGINGFACE_TOKEN is not defined
            pass

    if token is None:
        token = os.getenv("REQUIRES_HUGGINGFACE_TOKEN")

    if token is None:
        return True

    return bool(int(token))


def is_on_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ
