import sys
from pathlib import Path

# Ensure the repository root is importable so that `pipelines`,
# `p3_extreme_price_correction`, and `tests` packages can be imported from
# anywhere pytest is invoked (scoped to this tests/ directory).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
