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
        out_combined = os.path.join(temp_dir.name, "combined_graphs")
        out_validation = os.path.join(temp_dir.name, "validation_reports")

        # Ensure directories exist before running
        for path in [out_combined, out_validation]:
            os.makedirs(path, exist_ok=True)
        
        # Monkeypatch the module variables
        monkeypatch.setattr(rdf_main, "DIR_COMBINED", out_combined)
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


def test_instance_uris_are_scoped_per_dataset(monkeypatch):
    """Check that the same seed yields distinct deterministic URIs per dataset_id."""
    seed = rdf_main.build_organism_seed("female organism", "Taxon_A")

    monkeypatch.setitem(rdf_main.config, "dataset_id", "dataset-a")
    uri_a1 = rdf_main.generate_uri("org", seed)
    uri_a2 = rdf_main.generate_uri("org", seed)

    monkeypatch.setitem(rdf_main.config, "dataset_id", "dataset-b")
    uri_b = rdf_main.generate_uri("org", seed)

    assert uri_a1 == uri_a2  # deterministic within a dataset
    assert uri_a1 != uri_b   # distinct across datasets


def test_dataset_seed_salt_metadata_fallback_chain():
    """Check source_id/target_id fallback chain in the dataset salt."""
    build_salt = rdf_main.build_dataset_seed_salt
    cfg_blank = {"dataset_id": "", "input": {"json": "examples/minimal.json"}}

    def record(source_id="", target_id="", target_author=""):
        return {"C1": {"source_text": "x", "source_id": source_id,
                       "target_id": target_id, "target_author": target_author}}

    salt_full = build_salt(cfg=cfg_blank, metadata_map=record("paper-a", "orcid-1", "author-a"))
    salt_author = build_salt(cfg=cfg_blank, metadata_map=record("paper-a", "", "author-a"))
    salt_source_only = build_salt(cfg=cfg_blank, metadata_map=record("paper-a"))
    salt_no_source = build_salt(cfg=cfg_blank, metadata_map=record("", "orcid-1", "author-a"))

    assert salt_full != salt_author
    assert salt_author != salt_source_only
    assert salt_full != salt_source_only
    # Without source_id the record contributes nothing -> input-json fallback
    assert salt_no_source.startswith("input::")


def test_load_char_metadata_map_columns(tmp_path):
    """Check CSV column mapping into provenance records."""
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "Char_ID,Original_study_comment,Original_study_ID,Modelling_author,Modeller_ID\n"
        'C1,"Modified from character 7 of Some Study.","Some Study.","Author, A.","https://orcid.org/0000"\n',
        encoding="utf-8",
    )

    result = rdf_main.load_char_metadata_map(str(csv_path))

    assert result["C1"]["source_text"] == "character 7 of Some Study."
    assert result["C1"]["source_id"] == "Some Study."
    assert result["C1"]["target_author"] == "Author, A."
    assert result["C1"]["target_id"] == "https://orcid.org/0000"
