# SPDX-FileCopyrightText: 2025 Thiago S. R. Silva, Diego S. Porto
# SPDX-License-Identifier: MIT

import os
import tempfile
import importlib
import pytest
import yaml

# Load module properly
rdf_main = importlib.import_module("rdf_generator.main")

# Load configuration from project root
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Resolve input paths based on config
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", config["data_dir"])
INPUT_JSON = os.path.join(DATA_DIR, config["input"]["json"])
NEX_FILE = os.path.join(DATA_DIR, config["input"]["nex"])
SPECIES_FILE = os.path.join(DATA_DIR, config["input"]["species"])
SHACL_FILE = os.path.join(DATA_DIR, config["input"]["shacl"])


def test_inputs_exist():
    """Check that all input files exist."""
    missing = []
    for path, label in [
        (INPUT_JSON, "JSON"),
        (NEX_FILE, "NEX"),
        (SPECIES_FILE, "Species"),
        (SHACL_FILE, "SHACL"),
    ]:
        if not os.path.exists(path):
            missing.append(f"Missing {label}: {path}")
    if missing:
        pytest.skip("Skipping test due to missing inputs:\n" + "\n".join(missing))


def test_main_runs(monkeypatch):
    """
    Run the main function to ensure it doesn't crash.
    Redirect output directories to a temporary folder.
    """
    temp_dir = tempfile.TemporaryDirectory()
    try:
        # Define new temporary paths
        out_ttl = os.path.join(temp_dir.name, "output_ttl")
        out_png = os.path.join(temp_dir.name, "output_png")
        out_combined = os.path.join(temp_dir.name, "combined_graphs")
        out_graphviz = os.path.join(temp_dir.name, "graphviz_images")
        out_validation = os.path.join(temp_dir.name, "validation_reports")

        # Ensure directories exist before running
        for path in [out_ttl, out_png, out_combined, out_graphviz, out_validation]:
            os.makedirs(path, exist_ok=True)
        
        # Monkeypatch the module variables
        monkeypatch.setattr(rdf_main, "DIR_OUTPUT_TTL", out_ttl)
        monkeypatch.setattr(rdf_main, "DIR_OUTPUT_PNG", out_png)
        monkeypatch.setattr(rdf_main, "DIR_COMBINED", out_combined)
        monkeypatch.setattr(rdf_main, "DIR_GRAPHVIZ", out_graphviz)
        monkeypatch.setattr(rdf_main, "DIR_VALIDATION", out_validation)

        # Run the main() function — this should use the patched paths
        rdf_main.main()
    finally:
        temp_dir.cleanup()

def test_graph_building():
    """Check that base graph builds with expected namespaces."""
    from rdflib import Graph
    from rdf_generator.main import build_base_graph

    g = build_base_graph()
    expected_namespaces = [
        "bfo", "cdao", "dc", "dwc", "iao", "kb", "obo",
        "owl", "pato", "phb", "rdf", "rdfs", "ro", "txr", "uberon"
    ]
    found_ns = [prefix for prefix, _ in g.namespaces()]
    for ns in expected_namespaces:
        assert ns in found_ns, f"Namespace {ns} missing in base graph"
