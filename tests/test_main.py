# SPDX-FileCopyrightText: 2025 Thiago S. R. Silva, Diego S. Porto
# SPDX-License-Identifier: MIT

import os
import pytest
import yaml
from rdf_generator import main as rdf_main

# Load configuration
with open(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", config["data_dir"])
INPUT_JSON = os.path.join(DATA_DIR, config["input"]["json"])
NEX_FILE = os.path.join(DATA_DIR, config["input"]["nex"])
SPECIES_FILE = os.path.join(DATA_DIR, config["input"]["species"])
SHACL_FILE = os.path.join(DATA_DIR, config["input"]["shacl"])


def test_inputs_exist():
    """Check that all input files exist."""
    assert os.path.exists(INPUT_JSON), f"Missing input JSON: {INPUT_JSON}"
    assert os.path.exists(NEX_FILE), f"Missing NEX file: {NEX_FILE}"
    assert os.path.exists(SPECIES_FILE), f"Missing species file: {SPECIES_FILE}"
    assert os.path.exists(SHACL_FILE), f"Missing SHACL shapes: {SHACL_FILE}"


def test_main_runs(monkeypatch):
    """
    Run the main function to ensure it doesn't crash.
    Use monkeypatch to redirect output directories to a temp folder.
    """
    import tempfile

    temp_dir = tempfile.TemporaryDirectory()
    monkeypatch.setattr(rdf_main, "DIR_OUTPUT_TTL", os.path.join(temp_dir.name, "output_ttl"))
    monkeypatch.setattr(rdf_main, "DIR_OUTPUT_PNG", os.path.join(temp_dir.name, "output_png"))
    monkeypatch.setattr(rdf_main, "DIR_COMBINED", os.path.join(temp_dir.name, "combined_graphs"))
    monkeypatch.setattr(rdf_main, "DIR_GRAPHVIZ", os.path.join(temp_dir.name, "graphviz_images"))
    monkeypatch.setattr(rdf_main, "DIR_VALIDATION", os.path.join(temp_dir.name, "validation_reports"))

    # Should not raise
    rdf_main.main()
    temp_dir.cleanup()


def test_graph_building():
    """Check that base graph builds with expected namespaces."""
    from rdflib import Graph
    from rdf_generator.main import build_base_graph

    g = build_base_graph()
    expected_namespaces = ["bfo", "cdao", "dc", "dwc", "iao", "kb", "obo", "owl", "pato", "phb", "rdf", "rdfs", "ro", "txr", "uberon"]
    ns = [prefix for prefix, _ in g.namespaces()]
    for expected in expected_namespaces:
        assert expected in ns, f"Namespace {expected} missing in base graph"
