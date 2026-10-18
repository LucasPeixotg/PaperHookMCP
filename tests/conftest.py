"""Switch the paper cache off before `paperhook.config` is first imported.

The client tests count requests on a mocked transport, and a store left behind by a
real run would answer some of them. Tests of the cache build their own store under
`tmp_path`.

`paperhook` itself is importable because pyproject.toml sets pytest's `pythonpath`
to the repo root, so the suite runs without installing the package.
"""

import os

os.environ["CACHE_ENABLED"] = "false"
