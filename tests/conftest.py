"""Put `src/` on sys.path.

Imports in this repo are rooted at src/ (`from config import settings`), and
there is no pyproject.toml to install the package from, so the tests reproduce
what running `python src/server.py` does.

Also switches the paper cache off before `config` is first imported: the client tests
count requests on a mocked transport, and a store left behind by a real run would
answer some of them. Tests of the cache build their own store under `tmp_path`.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ["CACHE_ENABLED"] = "false"
