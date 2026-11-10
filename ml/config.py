"""
NeuroFusion AI — ML Config Forwarder
Re-exports root configuration for backward compatibility within ML modules.
"""
import os, sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import *
