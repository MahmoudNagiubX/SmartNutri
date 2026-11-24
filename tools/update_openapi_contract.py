"""Regenerate the checked-in OpenAPI v1 contract from the current FastAPI app."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smartnutri_backend.api import create_app
from smartnutri_backend.settings import Settings


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="smartnutri-openapi-") as temporary_data:
        with TestClient(
            create_app(
                Settings(
                    data_dir=Path(temporary_data),
                    scale_mode="mock",
                    cors_origins=(),
                )
            )
        ) as client:
            schema = client.get("/openapi.json").json()
    output = ROOT / "docs" / "contracts" / "openapi-v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(schema['paths'])} API paths to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
