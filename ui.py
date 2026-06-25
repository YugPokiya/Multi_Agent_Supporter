"""Backward-compatible UI launcher for the FastAPI app.

Phase 2 moved the unstructured http.server implementation to FastAPI in
``app.main``. Run this file to start the same UI with Uvicorn.
"""


def run_ui(host: str = "127.0.0.1", port: int = 8081) -> None:
    import uvicorn

    uvicorn.run("app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_ui()
