# SPDX-FileCopyrightText: 2025 Thiago S. R. Silva, Diego S. Porto
# SPDX-License-Identifier: MIT

import os
import json
import yaml
import uuid
import re
from typing import Optional, Tuple, Dict, Any, List

import dendropy
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL
from pyshacl import validate
from pygraphviz import AGraph

# === CONFIGURATION ===
# Load configuration from YAML file
def load_config(config_path=os.path.join("configs", "config.yaml")):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load the YAML config
config = load_config()

# BASE_DIR points to /app inside container
BASE_DIR = "/app"

# DATA_DIR from config
DATA_DIR = os.path.join(BASE_DIR, config["data_dir"])

# OUTPUT_DIR from config
OUTPUT_DIR = config["output"]["base_dir"]

# Input files
INPUT_JSON = os.path.join(DATA_DIR, config["input"]["json"])
NEX_FILE = os.path.join(DATA_DIR, config["input"]["nex"])
SPECIES_FILE = os.path.join(DATA_DIR, config["input"]["species"])
SHACL_FILE = os.path.join(DATA_DIR, config["input"]["shacl"])

# Output directories
DIR_OUTPUT_TTL = os.path.join(OUTPUT_DIR, config["output"]["ttl"])
DIR_OUTPUT_PNG = os.path.join(OUTPUT_DIR, config["output"]["png"])
DIR_VALIDATION = os.path.join(OUTPUT_DIR, config["output"]["validation"])
DIR_COMBINED = os.path.join(OUTPUT_DIR, config["output"]["combined"])
DIR_GRAPHVIZ = os.path.join(OUTPUT_DIR, config["output"]["graphviz"])

# === SETUP ===
# Create output dirs if they don’t exist
for d in [DIR_OUTPUT_TTL, DIR_OUTPUT_PNG, DIR_VALIDATION, DIR_COMBINED, DIR_GRAPHVIZ]:
    os.makedirs(d, exist_ok=True)

# Reset summary each run
with open(os.path.join(DIR_VALIDATION, "validation_summary.txt"), "w", encoding="utf-8") as f:
    f.write("")

# === NAMESPACES ===
BFO = Namespace("http://purl.obolibrary.org/obo/BFO_")
CDAO = Namespace("http://purl.obolibrary.org/obo/CDAO_")
DC = Namespace("http://purl.org/dc/terms#")
DWC = Namespace("http://rs.tdwg.org/dwc/terms#")
IAO = Namespace("http://purl.obolibrary.org/obo/IAO_")
KB = Namespace("http://www.phenobees.org/kb#")
OBO = Namespace("http://purl.obolibrary.org/obo#")
PATO = Namespace("http://purl.obolibrary.org/obo/PATO_")
PHB = Namespace("http://www.phenobees.org/ontology#")
RO = Namespace("http://purl.obolibrary.org/obo/RO_")
TXR = Namespace("http://purl.obolibrary.org/obo/TAXRANK_")
UBERON = Namespace("http://purl.obolibrary.org/obo/UBERON_")
UUID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

# === GLOBAL MAPS ===
CHAR_STATE_MAPPING: Dict[str, Dict[int, str]] = {}   # Char_ID -> {state_index -> state_uri_str}
CHAR_URI_BY_ID: Dict[str, URIRef] = {}               # Char_ID -> character URI

# === HELPERS ===

# ---------- Graph setup helpers ----------
def bind_namespaces(g: Graph) -> Graph:
    """
    Bind all commonly used ontology namespaces to the given RDFLib graph.

    Ensures a consistent prefix ordering so Turtle serialization is stable.

    Args:
        g: RDFLib Graph to bind namespaces to.

    Returns:
        The same Graph instance with namespaces bound.
    """
    ordered_prefixes = [
        ("bfo", BFO),
        ("cdao", CDAO),
        ("dc", DC),
        ("dwc", DWC),
        ("iao", IAO),
        ("kb", KB),
        ("obo", OBO),
        ("owl", OWL),
        ("pato", PATO),
        ("phb", PHB),
        ("rdf", RDF),
        ("rdfs", RDFS),
        ("ro", RO),
        ("txr", TXR),
        ("uberon", UBERON)
    ]
    for prefix, ns in ordered_prefixes:
        g.bind(prefix, ns, replace=True)

    return g

def build_base_graph() -> Graph:
    """
    Create a base RDF graph with ontology class scaffolding.

    Includes:
      - Minimal UBERON classes (adult, female, male organisms).
      - Core PHB statement classes (neomorphic, transformational).
      - Proper rdfs:label annotations for readability.

    Returns:
        A new RDFLib Graph pre-populated with base ontology classes.
    """
    base = Graph()
    bind_namespaces(base)

    # minimal class declarations used across character graphs
    base.add((UBERON.UBERON_0007023, RDF.type, OWL.Class))
    base.add((UBERON.UBERON_0003100, RDF.type, OWL.Class))
    base.add((UBERON.UBERON_0003101, RDF.type, OWL.Class))

    base.add((UBERON.UBERON_0007023,RDFS.label, Literal("Adult Organism")))
    base.add((UBERON.UBERON_0003100,RDFS.label, Literal("Female Organism")))
    base.add((UBERON.UBERON_0003101,RDFS.label, Literal("Male Organism")))

    base.add((PHB.NeomorphicStatement, RDF.type, OWL.Class))
    base.add((PHB.TransformationalSimpleStatement, RDF.type, OWL.Class))
    base.add((PHB.TransformationalComplexStatement, RDF.type, OWL.Class))

    base.add((PHB.NeomorphicStatement, RDFS.label, Literal("Neomorphic Statement")))
    base.add((PHB.TransformationalSimpleStatement, RDFS.label, Literal("Transformational Simple Statement")))
    base.add((PHB.TransformationalComplexStatement, RDFS.label, Literal("Transformational Complex Statement")))
    
    return base

# ---------- Load inputs (data, matrix, shapes) ----------

print("\n=== Loading characters ===")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    raw = json.load(f)
    dataset: List[Dict[str, Any]] = raw if isinstance(raw, list) else [raw]

# --- Normalize locators so each is a dict ---
for row in dataset:
    locators = row.get("Locators", [])
    normalized_locators = []
    for loc in locators:
        if isinstance(loc, dict):
            normalized_locators.append(loc)
        elif isinstance(loc, str):
            normalized_locators.append({"label": loc.split("/")[-1], "uri": loc})
        elif isinstance(loc, URIRef):
            normalized_locators.append({"label": str(loc).split("/")[-1], "uri": str(loc)})
        else:
            print(f"[WARN] Unexpected locator type {type(loc)} in row {row.get('Char_ID')}")
    row["Locators"] = normalized_locators

    # Also recursively check if nested fields like Variable have locators
    var = row.get("Variable")
    if var and "Locators" in var:
        nested_locators = var["Locators"]
        normalized_nested = []
        for loc in nested_locators:
            if isinstance(loc, dict):
                normalized_nested.append(loc)
            elif isinstance(loc, str):
                normalized_nested.append({"label": loc.split("/")[-1], "uri": loc})
            elif isinstance(loc, URIRef):
                normalized_nested.append({"label": str(loc).split("/")[-1], "uri": str(loc)})
        var["Locators"] = normalized_nested

print("\n=== Loading species names ===")
with open(SPECIES_FILE, "r", encoding="utf-8") as f:
    species_list = json.load(f)
    species_data: Dict[str, Dict[str, Any]] = {s["input_species_name"]: s for s in species_list}

print("\n=== Loading NEXUS Matrix ===")
nexus_dataset = dendropy.DataSet.get(path=NEX_FILE, schema="nexus")
char_matrix = nexus_dataset.char_matrices[0]  # DendroPy CharacterMatrix

print("\n=== Loading SHACL Shapes ===")
shapes_graph = Graph()
bind_namespaces(shapes_graph)
shapes_graph.parse(SHACL_FILE, format="turtle")

# Build a normalized label -> URI lookup to assist negation/complements
state_label_to_uri: Dict[str, str] = {}
for entry in dataset:
    for s in entry.get("States", []):
        lab = next((v for k, v in s.items() if 'label' in k.lower()), "").strip().lower()
        u = next((v for k, v in s.items() if 'uri' in k.lower() and v), None)
        if lab and u and lab not in state_label_to_uri:
            state_label_to_uri[lab] = u

# ---------- Species processing helper ----------
def handle_species(
    sp_g: Graph,
    sp_label: str,
    sp_data: Dict[str, Any],
    species_data: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Any], URIRef, URIRef]:
    """
    Create RDF representations for a species concept and its instance.

    If the species is not already in `species_data`, a new concept and
    individual are added to the graph.

    Args:
        sp_g: Target RDFLib graph to add triples into.
        sp_label: Species name string from dataset row.
        sp_data: Raw species record (possibly incomplete).
        species_data: Global dictionary of known species metadata.

    Returns:
        A tuple of:
          - species_info: Consolidated species metadata dictionary.
          - sp_uri: URIRef for the species concept (OWL Class).
          - sp_instance: URIRef for the species instance individual.

    NOTE:
        Assumes each species is unique by its label string.
    """

    # --- Resolve species info ---
    species_info = {}
    for info in species_data.values():
        if info.get("valid_species_name") == sp_label or info.get("input_species_name") == sp_label:
            species_info = info.copy()  # make a copy to safely modify
            break
    if not species_info:
        species_info = sp_data.copy() if sp_data else {}

    # Ensure IDs are included
    if sp_data:
        if "ID" not in species_info and sp_data.get("ID"):
            species_info["ID"] = sp_data["ID"]
        if "zoobank_identifier" not in species_info and sp_data.get("zoobank_identifier"):
            species_info["zoobank_identifier"] = sp_data["zoobank_identifier"]

    # --- Species concept (class) ---
    sp_uri_str = species_info.get("URI") or str(KB[sp_label.replace(" ", "_")])
    sp_uri = URIRef(sp_uri_str)
    sp_g.add((sp_uri, RDF.type, OWL.Class))
    sp_g.add((sp_uri, RDFS.label, Literal(sp_label)))

    # --- Species instance ---
    sp_uuid = uuid.uuid5(UUID_NAMESPACE, sp_label.strip().lower())
    sp_instance = URIRef(KB[f"sp-{sp_uuid}"])
    sp_g.add((sp_instance, RDF.type, TXR["0000006"]))  # species individual
    sp_g.add((sp_instance, RDFS.label, Literal(sp_label)))
    sp_g.add((sp_instance, IAO["0000219"], sp_uri))  # denotes the concept
    
    # If we have an external ID (GBIF), link it
    if species_info.get("ID"):
        gbif_uri = URIRef(f"https://www.gbif.org/species/{species_info['ID']}")
        sp_g.add((sp_instance, DWC.parentNameUsageID, gbif_uri))

    # Zoobank ID if available
    if species_info.get("zoobank_identifier"):
        sp_g.add((sp_instance, DWC.taxonID, Literal(species_info["zoobank_identifier"])))

    return species_info, sp_uri, sp_instance

# ---------- Species graph serialization ----------

def serialize_species_graph(
        sp_g: Graph,
        sp_label: str,
        output_dir: str
    ) -> None:
    """
    Serialize the species graph into its own TTL file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a safe filename from species label
    safe_label = sp_label.replace(" ", "_").replace("/", "_")
    file_path = os.path.join(output_dir, f"{safe_label}.ttl")

    sp_g.serialize(destination=file_path, format="turtle")
    print(f"[Species TTL] Wrote {file_path}")


# ---------- Character-building helpers ----------

def handle_character_type(
        g: Graph,
        character: URIRef,
        data: Dict[str, Any]
    ) -> None:
    """
    Add RDF triples for the character type based on the presence of 'Variable'.
    Defensive against missing or None variable sections.
    """
    variable_data = data.get("Variable")
    if not variable_data:
        g.add((character, RDF.type, PHB.NeomorphicStatement))
        g.add((character, RDF.type, OWL.NamedIndividual))
        print(f"[Neomorphic] Char_ID: {data['Char_ID']}")
        return

    if variable_data.get("Variable comment"):
        g.add((character, RDF.type, PHB.TransformationalComplexStatement))
        g.add((character, RDF.type, OWL.NamedIndividual))
        print(f"[Transformational Complex] Char_ID: {data['Char_ID']}")
    else:
        g.add((character, RDF.type, PHB.TransformationalSimpleStatement))
        g.add((character, RDF.type, OWL.NamedIndividual))
        print(f"[Transformational Simple] Char_ID: {data['Char_ID']}")

def handle_organism(
        g: Graph,
        org_data: Dict[str, Any]
    ) -> URIRef:
    """
    Add RDF triples for the organism. Returns the organism instance URI.
    """
    org_label = org_data.get("Label", "Unknown organism")
    org_uri_str = org_data.get("URI") or str(KB[org_label.replace(" ", "_")])
    org_uri = URIRef(org_uri_str)

    org_uuid = uuid.uuid5(UUID_NAMESPACE, f"{org_label.strip().lower()}")
    instance_uri = URIRef(KB[f"org-{org_uuid}"])

    g.add((org_uri, RDFS.label, Literal(org_label)))
    g.add((org_uri, RDF.type, OWL.Class))

    g.add((instance_uri, RDFS.label, Literal(org_label)))
    g.add((instance_uri, RDF.type, org_uri))
    g.add((instance_uri, RDF.type, OWL.NamedIndividual))

    return instance_uri

def handle_locator(
    g: Graph,
    locator: Any,              
    parent_instance: URIRef    # now always organism or previous locator
) -> Optional[URIRef]:
    """
    Add RDF triples for an anatomical or other locator entity.

    Normalizes locator input into a dict (label + URI),
    creates both a class (if URI provided) and an instance,
    and chains the instance to its parent using BFO:0000051 (has_part).

    Args:
        g: RDFLib graph to populate.
        locator: Locator record (dict, str, or URIRef).
        parent_instance: The instance this locator belongs to.

    Returns:
        URIRef of the created locator instance, or None if malformed.

    NOTE:
        If `locator` is just a string, a new PHB class is created.
        If it's a dict, expected keys are {"label": str, "uri": str}.
    """
    # Normalize locator
    if isinstance(locator, dict):
        loc_dict = locator
    elif isinstance(locator, (str, URIRef)):
        loc_dict = {"label": str(locator).split("/")[-1], "uri": str(locator)}
    else:
        print(f"[WARN] Unexpected locator type {type(locator)}")
        return None

    label = next((v for k, v in loc_dict.items() if "label" in k.lower()), None)
    if not label:
        return None  # malformed entry

    uri = next((v for k, v in loc_dict.items() if "uri" in k.lower() and v), None)
    loc_uuid = uuid.uuid5(UUID_NAMESPACE, f"{uri or label.strip().lower()}")
    current_instance = URIRef(KB[f"loc-{loc_uuid}"])

    if uri:
        class_uri = URIRef(uri)
        g.add((class_uri, RDFS.label, Literal(label)))
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((current_instance, RDFS.label, Literal(label)))
        g.add((current_instance, RDF.type, class_uri))
    else:
        g.add((current_instance, RDFS.label, Literal(label)))

    g.add((current_instance, RDF.type, OWL.NamedIndividual))

    # --- Chain anatomy ---
    g.add((parent_instance, BFO["0000051"], current_instance))  # previous → has_part → current

    return current_instance

def handle_organism_and_locators(
    g: Graph,
    data: Dict[str, Any]
) -> Tuple[Optional[URIRef], List[URIRef]]:
    """
    Wrapper: handle organism + its locators.
    Returns a tuple:
        - organism_instance (or None)
        - list of locator instances in their chained order
    """

    organism_instance = handle_organism(g, data.get("Organism") or {})
    previous_instance = organism_instance
    locators: List[URIRef] = []

    for locator in data.get("Locators", []) or []:
        # Ensure locator is a dict
        if isinstance(locator, str) or isinstance(locator, URIRef):
            locator = {"label": str(locator).split("/")[-1], "uri": str(locator)}
        elif not isinstance(locator, dict):
            print(f"[WARN] Unexpected locator type {type(locator)} for Char_ID {data.get('Char_ID')}")
            continue

        current_instance = handle_locator(g, locator, previous_instance)
        if current_instance:
            locators.append(current_instance)
            previous_instance = current_instance  # chain continues

    return organism_instance, locators

def compute_default_organism_instance_uri_from_dataset(
        dataset: List[Dict[str, Any]]
    ) -> Optional[URIRef]:
    """
    Compute the canonical organism instance URI deterministically from the dataset.
    Uses the first row that has an Organism section. No triples are added here.
    """
    for row in dataset:
        org = row.get("Organism") or {}
        org_label = org.get("Label")
        if org_label:
            org_uuid = uuid.uuid5(UUID_NAMESPACE, f"{org_label.strip().lower()}")
            return URIRef(KB[f"org-{org_uuid}"])
    return None

def handle_variable_component(
    g: Graph,
    data: Dict[str, Any],
    final_component: Optional[URIRef] = None
) -> Optional[URIRef]:
    """
    Add RDF triples for the 'Variable' section. Returns the variable instance URIRef or None.
    """
    var_data = data.get("Variable")
    if not var_data:
        return final_component

    var_label = var_data.get("Variable label", "Unnamed Variable")
    var_uuid = uuid.uuid5(UUID_NAMESPACE, f"{var_data.get('Variable URI', var_label.strip().lower())}")
    var_instance_uri = URIRef(KB[f"var-{var_uuid}"])
    
    g.add((var_instance_uri, RDFS.label, Literal(var_label)))
    g.add((var_instance_uri, RDF.type, OWL.NamedIndividual))

    if var_data.get("Variable URI"):
        class_uri = URIRef(var_data["Variable URI"])
        g.add((var_instance_uri, RDF.type, class_uri))
        g.add((class_uri, RDFS.label, Literal(var_label)))

    if var_data.get("Variable comment"):
        g.add((var_instance_uri, RDFS.comment, Literal(var_data["Variable comment"])))

    return var_instance_uri

def handle_states(
    g: Graph,
    # char_uri: URIRef,
    data: Dict[str, Any],
    final_component: Optional[URIRef] = None
) -> Dict[int, str]:
    """
    Add RDF triples for 'States'. Returns a map of index -> state node URI (str).
    Negations like "not X" or "absent X" are normalized into positive
    absence-style labels (e.g., "X absent"), avoiding owl:complementOf.
    """
    state_map_for_char: Dict[int, str] = {}

    for state_index, state in enumerate(data.get("States", []) or []):
        label = next((v for k, v in state.items() if 'label' in k.lower()), "unknown").strip()
        uri = next((v for k, v in state.items() if 'uri' in k.lower() and v), None)

        normalized_label = label.lower()
        # Detect negations
        if normalized_label.startswith("not "):
            base_label = label[4:].strip()
            label = f"not {base_label}"

        # UUID for state
        state_uuid = uuid.uuid5(UUID_NAMESPACE, f"{data['Char_ID']}_{uri or label.lower()}")
        state_node = URIRef(KB[f"sta-{state_uuid}"])

        g.add((state_node, RDF.type, CDAO["0000045"]))  # CDAO State

        # Link to final_component from variable ---
        if final_component:
            g.add((final_component, PHB.has_characteristic, state_node))

        # Type assignment
        g.add((state_node, RDFS.label, Literal(label)))
        if uri:
            g.add((state_node, RDF.type, URIRef(uri)))
        g.add((state_node, RDF.type, OWL.NamedIndividual))

        # Link state to character
        state_map_for_char[state_index] = str(state_node)

    return state_map_for_char

def process_phenotype(g: Graph, row: Dict[str, Any]) -> Tuple[URIRef, URIRef, Dict[int, str], Graph]:
    """
    Construct a phenotype statement graph for a single dataset row.

    Workflow:
      1. Create a Character (CDAO:Character) and label it.
      2. Create a Phenotype statement individual with correct type.
      3. Add organism, locators, variable component, and states.
      4. Optionally build a minimal species graph.

    Args:
        g: Graph into which phenotype triples are added.
        row: Dataset row containing phenotype definition.

    Returns:
        A tuple of:
          - char_uri: URIRef for the Character class.
          - phenotype_instance: URIRef for the phenotype statement individual.
          - state_map_for_char: Mapping of state indices to KB URIs.
          - sp_g: Species-specific RDFLib Graph (possibly empty).
    """
    char_id = row.get("CharacterID") or str(uuid.uuid4())
    char_label = row.get("CharacterLabel", f"Character {char_id}")

    # Character class definition
    char_uuid = uuid.uuid5(UUID_NAMESPACE, f"char_{char_id}")
    char_uri = URIRef(KB[f"char-{char_uuid}"])
    g.add((char_uri, RDF.type, CDAO["0000075"]))  # CDAO Character
    g.add((char_uri, RDFS.label, Literal(char_label)))

    # Phenotype instance
    pheno_uuid = uuid.uuid5(UUID_NAMESPACE, f"pheno_{char_id}")
    phenotype_instance = URIRef(KB[f"phe-{pheno_uuid}"])
    g.add((phenotype_instance, RDF.type, OWL.NamedIndividual))
    g.add((phenotype_instance, RDFS.label, Literal(f"Phenotype statement for {char_label}")))

    # Decide statement class based on Variable section
    variable_data = row.get("Variable")
    if not variable_data:
        g.add((phenotype_instance, RDF.type, PHB.NeomorphicStatement))
    elif variable_data.get("Variable comment"):
        g.add((phenotype_instance, RDF.type, PHB.TransformationalComplexStatement))
    else:
        g.add((phenotype_instance, RDF.type, PHB.TransformationalSimpleStatement))

    # Organism + locators
    organism_instance, locator_instances = handle_organism_and_locators(g, row)
    if organism_instance:
        g.add((phenotype_instance, PHB.has_organismal_component, organism_instance))
    for locator in locator_instances:
        g.add((phenotype_instance, PHB.has_entity_component, locator))

    # Variable Component
    variable_instance = handle_variable_component(g, row, final_component=phenotype_instance)
    if variable_instance:
        g.add((phenotype_instance, PHB.has_variable_component, variable_instance))

    # States / qualities
    state_map_for_char = handle_states(g, row, final_component=phenotype_instance)
    for idx, state_uri in state_map_for_char.items():
        g.add((phenotype_instance, PHB.has_quality_component, URIRef(state_uri)))

        # === NEW: link Character → may_have_state → State ===
        g.add((char_uri, CDAO.may_have_state, URIRef(state_uri)))
        print(f"[DEBUG] {char_label} (ID {char_id}) may_have_state -> {state_uri}")

    # Species Graph
    sp_g = Graph()
    sp_label = row.get("SpeciesLabel")
    species_id = row.get("SpeciesID")
    if sp_label and species_id:
        sp_uri = URIRef(KB[f"sp-{uuid.uuid5(UUID_NAMESPACE, sp_label)}"])
        sp_g.add((sp_uri, RDF.type, PHB.Species))
        sp_g.add((sp_uri, RDFS.label, Literal(sp_label)))
        sp_g.add((sp_uri, DWC.parentNameUsageID, URIRef(f"https://www.gbif.org/species/{species_id}")))
        print(f"[DEBUG] Species graph for {sp_label} has {len(sp_g)} triples:")
        for s, p, o in sp_g:
            print(f"  {s} {p} {o}")

    return char_uri, phenotype_instance, state_map_for_char, sp_g

# ---------- Validation + Serialization ----------

def perform_shacl_validation(g: Graph, shapes: Graph) -> Tuple[bool, Graph, str]:
    return validate(data_graph=g, shacl_graph=shapes, inference='rdfs')

def write_validation_results(
    entity_id: str,
    conforms: bool,
    results_graph: Graph,
    results_text: str,
    combined_report_graph: Graph,
    validation_dir: str
) -> None:
    if conforms:
        print(f"[VALID] SHACL validation passed for {entity_id}")
    else:
        print(f"[INVALID] SHACL validation failed for {entity_id}")
        print(results_text)

    text_report_path = os.path.join(validation_dir, "validation_summary.txt")
    with open(text_report_path, "a", encoding="utf-8") as text_file:
        text_file.write(f"==== {entity_id} ====\n")
        text_file.write(results_text)
        text_file.write("\n\n")

    for triple in results_graph:
        combined_report_graph.add(triple)

def validate_graph_and_record(
    entity_id: str,
    g: Graph,
    shapes: Graph,
    combined_report_graph: Graph,
    validation_dir: str
) -> bool:
    conforms, results_graph, results_text = perform_shacl_validation(g, shapes)
    write_validation_results(entity_id, conforms, results_graph, results_text,
                             combined_report_graph, validation_dir)
    return conforms

def write_ttl_with_sections(graph: Graph, ttl_file: str) -> None:
    """Write TTL grouped into Classes, Individuals (grouped), Properties, and Other."""
    # Serialize once to get the prefix block
    ttl_full = graph.serialize(format="turtle", encoding="utf-8").decode("utf-8")
    prefix_block = ttl_full.split("\n\n", 1)[0]

    def _write_triple(f, s, p, o):
        if isinstance(o, URIRef):
            f.write(f"<{s}> <{p}> <{o}> .\n")
        else:
            # Literal or BNode
            f.write(f"<{s}> <{p}> {o.n3()} .\n")

    # Derive local namespace string for "local class used as rdf:type object" fallback
    KB_NS = str(KB)

    # Collect sets
    class_nodes = set(graph.subjects(RDF.type, OWL.Class)) | set(graph.subjects(RDF.type, RDFS.Class))
    # Also include *local* URIs that are used as rdf:type objects (even if owl:Class not asserted)
    class_nodes |= {
        o for (_, _, o) in graph.triples((None, RDF.type, None))
        if isinstance(o, URIRef) and str(o).startswith(KB_NS)
    }

    individual_nodes = set(graph.subjects(RDF.type, OWL.NamedIndividual))

    object_properties = set(graph.subjects(RDF.type, OWL.ObjectProperty))
    data_properties   = set(graph.subjects(RDF.type, OWL.DatatypeProperty))
    annot_properties  = set(graph.subjects(RDF.type, OWL.AnnotationProperty))

    # For "Other", we’ll exclude anything whose subject is in any of the buckets above
    excluded_subjects = class_nodes | individual_nodes | object_properties | data_properties | annot_properties

    # DEBUG counts (optional — comment out if noisy)
    print("[DEBUG] Classes (owl/rdfs + local rdf:type objects):", len(class_nodes))
    print("[DEBUG] Individuals (owl:NamedIndividual):", len(individual_nodes))
    print("[DEBUG] ObjectProperties:", len(object_properties))
    print("[DEBUG] DatatypeProperties:", len(data_properties))
    print("[DEBUG] AnnotationProperties:", len(annot_properties))

    with open(ttl_file, "w", encoding="utf-8") as f:
        f.write(prefix_block + "\n\n")

        # === Classes ===
        f.write("### ==============================\n### Classes\n### ==============================\n\n")
        for s in sorted(class_nodes, key=lambda x: str(x)):
            for p, o in graph.predicate_objects(s):
                _write_triple(f, s, p, o)
            f.write("\n")

        # === Individuals (grouped by KB prefixes) ===
        f.write("### ==============================\n### Individuals (grouped)\n### ==============================\n\n")

        # Group by prefix buckets
        buckets = {
            "## --- KB#CHAR instances ---": lambda u: str(u).startswith(f"{KB_NS}char-"),
            "## --- KB#PHE instances ---":  lambda u: str(u).startswith(f"{KB_NS}phe-"),
            "## --- KB#TU instances ---":   lambda u: str(u).startswith(f"{KB_NS}tu-"),
            "## --- KB#LOC instances ---":  lambda u: str(u).startswith(f"{KB_NS}loc-"),
            "## --- KB#VAR instances ---":  lambda u: str(u).startswith(f"{KB_NS}var-"),
            "## --- KB#STA (state) ---":    lambda u: str(u).startswith(f"{KB_NS}sta-"),
            "## --- KB#SP (species inst) ---": lambda u: str(u).startswith(f"{KB_NS}sp-"),
            "## --- Other Individuals ---":  lambda u: True,  # fallback
        }

        remaining = set(individual_nodes)
        for header, pred in buckets.items():
            bucket_nodes = [u for u in remaining if pred(u)]
            if not bucket_nodes:
                continue
            f.write(header + "\n\n")
            for s in sorted(bucket_nodes, key=lambda x: str(x)):
                for p, o in graph.predicate_objects(s):
                    _write_triple(f, s, p, o)
                f.write("\n")
            # Remove written nodes so they don't show again
            remaining -= set(bucket_nodes)

        # === Properties ===
        f.write("### ==============================\n### Properties\n### ==============================\n\n")

        def write_prop_section(title: str, nodes: set):
            if not nodes:
                return
            f.write(title + "\n\n")
            for s in sorted(nodes, key=lambda x: str(x)):
                for p, o in graph.predicate_objects(s):
                    _write_triple(f, s, p, o)
                f.write("\n")

        write_prop_section("## --- ObjectProperties ---", object_properties)
        write_prop_section("## --- DatatypeProperties ---", data_properties)
        write_prop_section("## --- AnnotationProperties ---", annot_properties)

        # === Other Triples ===
        f.write("### ==============================\n### Other Triples\n### ==============================\n\n")
        for s, p, o in graph:
            if s in excluded_subjects:
                continue
            _write_triple(f, s, p, o)

def visualize_graph(classes, individuals, edges, output_file="graph.svg"):
    g = AGraph(directed=True, strict=False, rankdir="LR")
    g.node_attr.update(
        shape="ellipse", style="filled", fillcolor="#d5f5e3",
        margin="0.1,0.1", width="0.2", height="0.2",
        nodesep="1.0", ranksep="2.0", splines="true"
    )
    for cls in classes:
        g.add_node(cls, shape="box", fillcolor="#cce5ff")
    for ind in individuals:
        short_label = ind.split("#")[-1]
        g.add_node(ind, label=short_label)
    for src, dst, label in edges:
        g.add_edge(src, dst, label=label)
    if classes:
        g.add_subgraph(classes, rank='same')
    if individuals:
        g.add_subgraph(individuals, rank='same')
    g.layout(prog="fdp")
    g.draw(output_file)

# ---------- High-level build steps ----------

def build_character_graphs(
    dataset: List[Dict[str, Any]],
    base_graph: Graph,
    shapes: Optional[Graph] = None,
    combined_report_graph: Optional[Graph] = None,
    validation_dir: Optional[str] = None
) -> Tuple[Graph, Dict[str, Graph], Dict[str, Dict[int, str]], List[str]]:
    combined_char_graph = Graph()
    bind_namespaces(combined_char_graph)
    for t in base_graph:
        combined_char_graph.add(t)

    character_graphs: Dict[str, Graph] = {}
    char_state_mapping: Dict[str, Dict[int, str]] = {}
    char_ids_in_order: List[str] = []

    for row in dataset:
        char_id = row["Char_ID"]
        char_ids_in_order.append(char_id)

        g_char = Graph()
        bind_namespaces(g_char)
        for t in base_graph:
            g_char.add(t)

        # Build phenotype and species graph
        char_uri, phenotype_uri, state_map_for_char, sp_g = process_phenotype(g_char, row)
        char_state_mapping[char_id] = state_map_for_char

        # Merge species graph if present
        if sp_g and len(sp_g) > 0:
            g_char += sp_g
            combined_char_graph += sp_g
            print(f"[DEBUG] Merged sp_g with {len(sp_g)} triples into combined_char_graph")

        # Merge character graph
        combined_char_graph += g_char
        character_graphs[char_id] = g_char

        # Optional validation
        if shapes and combined_report_graph and validation_dir:
            validate_graph_and_record(
                entity_id=f"Char_ID {char_id}",
                g=g_char,
                shapes=shapes,
                combined_report_graph=combined_report_graph,
                validation_dir=validation_dir
            )

    print(f"[DEBUG] Total triples in combined_char_graph: {len(combined_char_graph)}")
    return combined_char_graph, character_graphs, char_state_mapping, char_ids_in_order

def build_cdao_matrix(
    nexus_matrix,
    dataset,
    char_ids_in_order: List[str]
) -> Tuple[Graph, URIRef]:
    """
    Build a CDAO matrix graph linking TUs, Characters, Cells, and Phenotypes.

    Args:
        nexus_matrix: DendroPy CharacterMatrix.
        dataset: List of dataset rows with characters and states.
        char_ids_in_order: Ordered list of character IDs matching matrix columns.

    Returns:
        A tuple of:
          - matrix_graph: RDFLib Graph of the matrix.
          - matrix_uri: URIRef of the matrix.
    """
    g = Graph()
    bind_namespaces(g)

    # Create matrix URI
    matrix_uri = URIRef(f"http://phenobees.org/kb#mx-{uuid.uuid4().hex[:8]}")
    g.add((matrix_uri, RDF.type, CDAO["0000056"]))  # CDAO Matrix
    g.add((matrix_uri, RDFS.label, Literal("matrix")))
    g.add((matrix_uri, DC.description, Literal("matrix description")))

    # TUs
    for taxon in nexus_matrix.taxon_namespace:
        tu_uuid = uuid.uuid5(UUID_NAMESPACE, taxon.label)
        tu_uri = URIRef(KB[f"tu-{tu_uuid}"])
        g.add((matrix_uri, CDAO["0000208"], tu_uri))  # has_TU

    # Characters (columns) + phenotypes + species
    for char_index, char_id in enumerate(char_ids_in_order):
        # Pull row from dataset
        char_data = next((row for row in dataset if row["Char_ID"] == char_id), None)
        if not char_data:
            continue  # skip if no matching row

        # Process phenotype
        char_uri, phenotype_uri, state_map, sp_g = process_phenotype(g, char_data)

        # Merge species triples
        if sp_g:
            g += sp_g

        # Link character into matrix
        g.add((matrix_uri, CDAO.has_character, char_uri))

        # Cells (taxon x character)
        for taxon in nexus_matrix.taxon_namespace:
            # UUID-based cell
            cell_uuid = uuid.uuid5(UUID_NAMESPACE, f"{taxon.label}_{char_index}")
            cell_uri = URIRef(KB[f"cell-{cell_uuid}"])

            # UUID-based TU (for consistency)
            tu_uuid = uuid.uuid5(UUID_NAMESPACE, taxon.label)
            tu_uri = URIRef(KB[f"tu-{tu_uuid}"])

            g.add((cell_uri, RDF.type, CDAO["0000008"]))  # CDAO Cell
            g.add((cell_uri, CDAO.belongs_to_TU, tu_uri))
            g.add((cell_uri, CDAO.belongs_to_character, char_uri))

            # Link Cell → Phenotype
            g.add((cell_uri, PHB.refers_to_phenotype_statement, phenotype_uri))

            # Link all states to the cell
            for state_uri_str in state_map.values():
                state_node = URIRef(state_uri_str)
                g.add((cell_uri, CDAO["0000184"], state_node))  # has_state

            # (state links per-taxon added later)

    return g, matrix_uri

# ---------- TU processing & outputs ----------

def enrich_and_serialize_tu_graph(
    tu_graph: Graph,
    tu_uri: URIRef,
    taxon_label: str,
    species_info: Dict[str, Any],
    sp_uri: URIRef,
    sp_instance: URIRef,
    instance_uri: URIRef,
    dir_combined: str
) -> None:
    """
    Add species info to a TU graph and serialize to TTL.

    Args:
        tu_graph: RDFLib Graph for the TU.
        tu_uri: URIRef of the TU individual.
        taxon_label: Label of the taxon.
        species_info: Dictionary with species metadata.
        sp_uri: URIRef of species concept.
        sp_instance: URIRef of species instance.
        instance_uri: Default organism instance URI.
        dir_combined: Output directory for TTL serialization.
    """
    # Format valid label
    valid_label = species_info.get("valid_species_name", taxon_label)
    parts = valid_label.split(" ", 2)
    binomial = f"{parts[0]} {parts[1]}" if len(parts) >= 2 else valid_label
    author = parts[2] if len(parts) == 3 else ""
    valid_label_html = f"<i>{binomial}</i> {author}".strip()

    # TU assertions
    tu_graph.add((tu_uri, RDF.type, OWL.NamedIndividual))
    tu_graph.add((tu_uri, RDFS.label, Literal(valid_label_html)))
    tu_graph.add((instance_uri, RO.has_role_in_modelling, tu_uri))
    tu_graph.add((tu_uri, RDF.type, CDAO["0000138"]))  # CDAO Taxon Unit
    tu_graph.add((tu_uri, IAO["0000219"], sp_instance)) # denotes

    # --- Connect TU to species concept as well ---
    tu_graph.add((sp_instance, IAO["0000219"], sp_uri))        # sp_instance denotes species class
    tu_graph.add((tu_uri, RO.denotes, sp_instance))                 # TU denotes species instance
    
    # Write TU graph to TTL
    ttl_file = os.path.join(dir_combined, f"tu_{taxon_label.replace(' ', '_')}.ttl")
    write_ttl_with_sections(tu_graph, ttl_file)
    print(f"[OK] TTL written for {taxon_label} → {ttl_file}")

def visualize_tu_graph(tu_graph: Graph, tu_uri: URIRef, output_dir: str, taxon_label: str) -> None:
    role_map = {}
    for s, p, o in tu_graph:
        if p == PHB.has_entity_component:
            role_map[o] = "locator"
        elif p == PHB.has_variable_component:
            role_map[o] = "variable"
        elif p == PHB.has_quality_component:
            role_map[o] = "state"
        elif p == PHB.has_organismal_component:
            role_map[o] = "organism"
        if s == tu_uri:
            role_map[s] = "TU"

    classes, individuals, edges = [], [], []
    all_nodes = set()
    for s, p, o in tu_graph:
        all_nodes.update([s, o])

    for n in all_nodes:
        role = role_map.get(n)
        n_str = str(n)
        if role in ("TU", "locator", "variable", "state", "organism"):
            individuals.append(n_str)
        else:
            classes.append(n_str)

    for s, p, o in tu_graph:
        edges.append((str(s), str(o), p.split('#')[-1]))

    svg_file = os.path.join(output_dir, f"tu_{taxon_label.replace(' ', '_')}.svg")
    visualize_graph(classes, individuals, edges, output_file=svg_file)
    print(f"[OK] {taxon_label} → SVG: {svg_file}")

def process_taxon(
    taxon,
    sp_g: Graph,
    species_data: Dict[str, Dict[str, Any]],
    matrix_graph: Graph,
    character_graphs: Dict[str, Graph],
    char_state_mapping: Dict[str, Dict[int, str]],
    char_ids_in_order: List[str],
    shapes: Graph,
    combined_report_graph: Graph,
    validation_dir: str,
    dir_combined: str,
    dir_graphviz: str
) -> Graph:  # return the TU graph so main can merge it later
    taxon_label = str(taxon.label)

    # Build a species subgraph for THIS taxon and get its instance + info
    sp_graph = Graph()
    bind_namespaces(sp_graph)

    # Get the best match dict for this taxon label (or an empty dict)
    sp_data_guess = species_data.get(taxon_label, {})
    species_info, sp_uri, sp_instance = handle_species(
        sp_g=sp_graph,
        sp_label=taxon_label,
        sp_data=sp_data_guess,
        species_data=species_data
    )

    # Serialize species graph to its own TTL file
    serialize_species_graph(sp_graph, taxon_label, os.path.join(dir_combined, "species"))

    # TU URI
    tu_uuid = uuid.uuid5(UUID_NAMESPACE, f"{taxon_label.strip().lower()}")
    tu_uri = URIRef(KB[f"tu-{tu_uuid}"])

    # Build a TU-specific graph
    tu_graph = Graph()
    bind_namespaces(tu_graph)

    # Copy relevant TU triples from matrix_graph
    tu_fragment = f"tu-{taxon_label.replace(' ', '_')}"
    for s, p, o in matrix_graph:
        if tu_fragment in str(s) or tu_fragment in str(o):
            tu_graph.add((s, p, o))

    # Merge per-character graphs and link states according to matrix
    for char_index, cell in enumerate(char_matrix[taxon]):
        if char_index >= len(char_ids_in_order):
            continue
        char_id = char_ids_in_order[char_index]
        g_char = character_graphs.get(char_id)

        # Merge the character graph (if built)
        if g_char:
            for t in g_char:
                tu_graph.add(t)

        state_symbol = str(cell).strip() if cell is not None else ""
        # Unknown state
        if state_symbol == '?':
            unknown_uuid = uuid.uuid5(UUID_NAMESPACE, f"{char_id}_unknown")
            unknown_node = URIRef(KB[f"unknown_state_{unknown_uuid}"])
            tu_graph.add((unknown_node, RDFS.label, Literal("unknown")))
            tu_graph.add((unknown_node, RDF.type, OWL.NamedIndividual))
            if g_char:
                pheno_seed = f"pheno_{char_id}"
                pheno_uuid = uuid.uuid5(UUID_NAMESPACE, pheno_seed)
                pheno_uri = URIRef(KB[f"phe-{pheno_uuid}"])
                tu_graph.add((pheno_uri, PHB.has_quality_component, unknown_node))
            continue

        # Skip polymorphic / missing states
        if not state_symbol or state_symbol == '-':
            continue
        try:
            state_index = int(state_symbol)
        except ValueError:
            continue

        # Link character to the resolved state node
        state_uri_str = char_state_mapping.get(char_id, {}).get(state_index)
        if state_uri_str and g_char:
            char_uri = URIRef(KB[f"char-{uuid.uuid5(UUID_NAMESPACE, str(char_id))}"])
            tu_graph.add((char_uri, PHB.may_have_state, URIRef(state_uri_str)))
            print(f"[TU mapping] {taxon_label} -> Char {char_id}, StateIndex {state_index}")

    # Merge species triples into the passed sp_g
    sp_g += sp_graph

    # Validate TU graph (centralized)
    validate_graph_and_record(
        entity_id=f"TU {taxon_label}",
        g=tu_graph,
        shapes=shapes,
        combined_report_graph=combined_report_graph,
        validation_dir=validation_dir
    )

    # Compute a stable organism instance URI for this TU
    instance_uri = compute_default_organism_instance_uri_from_dataset(dataset)  # uses global 'dataset'

    # Enrich + serialize TTL
    enrich_and_serialize_tu_graph(
        tu_graph,
        tu_uri,
        taxon_label,
        species_info,
        sp_uri,
        sp_instance,
        instance_uri,
        dir_combined
    )

    # Visualize TU
    visualize_tu_graph(tu_graph, tu_uri, dir_graphviz, taxon_label)

    return tu_graph  # return the TU graph for main to merge later

# ---------- MAIN Orchestration (single shared pipeline) ----------

def main():
    print("\n=== Building base graph and character graphs ===")
    base_graph = build_base_graph()

    # Combined report graph for all validations
    combined_report_graph = Graph()
    bind_namespaces(combined_report_graph)

    # Global species graph accumulator
    sp_g = Graph()
    bind_namespaces(sp_g)

    # 1) Build per-character graphs
    combined_char_graph, character_graphs, char_state_mapping, char_ids_in_order = build_character_graphs(
        dataset=dataset,
        base_graph=base_graph,
        shapes=shapes_graph,
        combined_report_graph=combined_report_graph,
        validation_dir=DIR_VALIDATION
    )

    print("\n=== Building CDAO matrix graph (shared pipeline) ===")
    matrix_graph, matrix_uri = build_cdao_matrix(
        nexus_matrix=char_matrix,
        dataset=dataset,
        char_ids_in_order=char_ids_in_order
    )

    print("\n=== Processing each taxon (TU graphs) ===")
    tu_graphs = []  # collect all TU graphs
    for taxon in char_matrix.taxon_namespace:
        g_tu = process_taxon(
            taxon=taxon,
            sp_g=sp_g,  # <<<<< pass the shared accumulator
            species_data=species_data,
            matrix_graph=matrix_graph,
            character_graphs=character_graphs,
            char_state_mapping=char_state_mapping,
            char_ids_in_order=char_ids_in_order,
            shapes=shapes_graph,
            combined_report_graph=combined_report_graph,
            validation_dir=DIR_VALIDATION,
            dir_combined=DIR_COMBINED,
            dir_graphviz=DIR_GRAPHVIZ
        )
        tu_graphs.append(g_tu)

    # Write individual components (optional)
    matrix_ttl = os.path.join(DIR_COMBINED, "matrix.ttl")
    write_ttl_with_sections(matrix_graph, matrix_ttl)

    chars_ttl = os.path.join(DIR_COMBINED, "characters_combined.ttl")
    write_ttl_with_sections(combined_char_graph, chars_ttl)

    species_ttl = os.path.join(DIR_COMBINED, "species_combined.ttl")
    write_ttl_with_sections(sp_g, species_ttl)

    # === Merge everything into one combined TTL ===
    final_graph = Graph()
    bind_namespaces(final_graph)

    # Merge base, characters, matrix, species, and TU graphs
    for t in base_graph:
        final_graph.add(t)
    for t in combined_char_graph:
        final_graph.add(t)
    for t in matrix_graph:
        final_graph.add(t)
    for t in sp_g:
        final_graph.add(t)
    for g_tu in tu_graphs:
        for t in g_tu:
            final_graph.add(t)

    merged_ttl = os.path.join(DIR_COMBINED, "all_combined.ttl")
    write_ttl_with_sections(final_graph, merged_ttl)
    print(f"[OK] Full combined TTL → {merged_ttl}")

if __name__ == "__main__":
    main()