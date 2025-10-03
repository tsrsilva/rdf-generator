# SPDX-FileCopyrightText: 2025 Thiago S. R. Silva, Diego S. Porto
# SPDX-License-Identifier: MIT

"""
RDF Generator
-------------
Generates RDF phenotype graphs from JSON and NEXUS matrices.
"""

__version__ = "0.0.1"  # initial version; semantic-release will auto-bump

# Import the main entry point for easy access
from .main import main

__all__ = ["main"]
