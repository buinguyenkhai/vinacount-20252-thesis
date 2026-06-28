"""Environment loading helpers for local runtime configuration."""

from __future__ import annotations


def load_dotenv_if_available() -> None:
    """Load a cwd-scoped .env file without overriding exported env vars."""
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
