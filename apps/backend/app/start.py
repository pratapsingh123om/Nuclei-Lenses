from __future__ import annotations

import os
from pathlib import Path

from .bootstrap import bootstrap_model


def main() -> None:
    model_dir = Path(os.getenv("NUCLEI_MODEL_DIR", "/app/model"))
    status = bootstrap_model(model_dir)
    print(f"Model bootstrap: {status}", flush=True)
    port = os.getenv("PORT", "8000")
    os.execvp(
        "uvicorn",
        ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", port],
    )


if __name__ == "__main__":
    main()

