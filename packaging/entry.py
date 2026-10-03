"""PyInstaller entry script (PyInstaller cannot start from `-m app.desktop`)."""
import sys

from app.desktop.__main__ import main

sys.exit(main())
