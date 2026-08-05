"""Disposable four-sample smoke run; never writes a formal checkpoint."""

import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multiscale_pcnss import RUN_CONFIG, run_smoke


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pcnss_smoke_") as directory:
        values = dict(
            RUN_CONFIG,
            stage="smoke_train",
            output_root=str(Path(directory) / "smoke"),
        )
        print(json.dumps(run_smoke(values), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
