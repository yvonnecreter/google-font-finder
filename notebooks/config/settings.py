import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATABASE = {
    "path": str(BASE_DIR / "db" / "fonts.db"),
}
DEBUG = False
DIR = {
    "fonts": str(BASE_DIR / "db" / "google-fonts"),
}