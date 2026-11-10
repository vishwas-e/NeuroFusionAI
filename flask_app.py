"""
NeuroFusion AI™ — Application Entry Point
==========================================
Launches the clinical workstation from the project root.
Delegates to the segregated backend application in `backend/flask_app.py`.

Run:
    python flask_app.py
    # or
    python run.py
"""

import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from backend.flask_app import app, main

if __name__ == "__main__":
    main()
