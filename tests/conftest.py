import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
# Both the bundled scripts and the shared recipe package, so a test module can
# import either one first without depending on which pulls the other in.
for _location in (_ROOT / "scripts", _ROOT / "shared"):
    if str(_location) not in sys.path:
        sys.path.insert(0, str(_location))
