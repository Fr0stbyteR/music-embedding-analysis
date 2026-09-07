from __future__ import annotations

import json
import secrets

import uvicorn

from .app import create_app
from .config import Settings


def main() -> None:
    settings = Settings()
    if settings.session_token is None:
        settings.session_token = secrets.token_urlsafe(32)
    settings.prepare()
    print(json.dumps({"protocol": "music-annotation/1", "host": settings.host, "port": settings.port, "token": settings.session_token}), flush=True)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level=settings.log_level)


if __name__ == "__main__":
    main()

