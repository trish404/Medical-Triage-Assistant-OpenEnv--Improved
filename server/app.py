"""
OpenEnv-compliant server entry point.
This module imports the FastAPI app from the root server.py file.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import from root server.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app

__all__ = ["app"]


def main():
    """Main entry point for the server."""
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()