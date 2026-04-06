"""Root-level server entry point — required by openenv validate."""

import sys
import os

# Allow importing ad_review_env when running from repo root without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ad_review_env"))

try:
    from ad_review_env.server.app import app
except ImportError:
    from server.app import app  # type: ignore[no-redef]


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
