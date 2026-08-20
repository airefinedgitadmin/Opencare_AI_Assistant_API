"""Uvicorn entrypoint.

Deliberately thin: the app itself is built by `create_app()` so it can be
exercised directly in tests via `TestClient(create_app())` without going
through uvicorn.
"""

import uvicorn

from careplan_ai.api.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("careplan_ai.main:app", host="0.0.0.0", port=8000, log_config=None)
