"""
FastVideo AI Studio Pro - Local Launch Interface
================================================
This entrypoint forwards directly to the core production studio module (studio.app)
providing the full multi-scene, batch queue, and super-resolution video studio.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run_studio import main

if __name__ == "__main__":
    main()
