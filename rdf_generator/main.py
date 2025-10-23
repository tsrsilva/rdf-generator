# SPDX-FileCopyrightText: 2025 Thiago S. R. Silva, Diego S. Porto
# SPDX-License-Identifier: MIT

import os
import json
import yaml
import uuid
from typing import Optional, Tuple, Dict, Any, List

import argparse
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
BASE_DIR = os.getcwd()

# DATA_DIR from config
DATA_DIR = os.path.join(BASE_DIR, config["data_dir"])

# OUTPUT_DIR from config
# OUTPUT_DIR = config["output"]["base_dir"]
output_base = config["output"]["base_dir"]
if os.path.isabs(output_base):
    OUTPUT_DIR = output_base  # absolute paths stay as-is
else:
    OUTPUT_DIR = os.path.join(BASE_DIR, output_base)

# Input files
INPUT_JSON = os.path.join(DATA_DIR, config["input"]["json"])
NEX_FILE = os.path.join(DATA_DIR, config["input"]["nex"])
SPECIES_FILE = os.path.join(DATA_DIR, config["input"]["species"])
SHACL_FILE = os.path.join(DATA_DIR, config["input"]["shacl"])

# Output directories
DIR_VALIDATION = os.path.join(OUTPUT_DIR, config["output"]["validation"])
DIR_COMBINED = os.path.join(OUTPUT_DIR, config["output"]["combined"])
DIR_GRAPHVIZ = os.path.join(OUTPUT_DIR, config["output"]["graphviz"])

# === SETUP ===
# Create output dirs if they don’t exist
for d in [DIR_VALIDATION, DIR_COMBINED, DIR_GRAPHVIZ]:
    os.makedirs(d, exist_ok=True)

# Reset summary each run
with open(os.path.join(DIR_VALIDATION, "validation_summary.txt"), "w", encoding="utf-8") as f:
    f.write("")

# === NAMESPACES ===
BFO = Namespace("http://purl.obolibrary.org/obo/BFO_")
CDAO = Namespace("http://purl.obolibrary.org/obo/CDAO_")
DC = Namespace("http://purl.org/dc/terms/")
DWC = Namespace("http://rs.tdwg.org/dwc/terms/")
IAO = Namespace("http://purl.obolibrary.org/obo/IAO_")
KB = Namespace("http://www.phenobees.org/kb#")
OBO = Namespace("http://purl.obolibrary.org/obo#")
PATO = Namespace("http://purl.obolibrary.org/obo/PATO_")
PHB = Namespace("http://www.phenobees.org/ontology#")
RO = Namespace("http://purl.obolibrary.org/obo/RO_")
TXR = Namespace("http://purl.obolibrary.org/obo/TAXRANK_")
UBERON = Namespace("http://purl.obolibrary.org/obo/UBERON_")
UUID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")

# === HELPERS ===

# ---------- Common utility functions ----------

def create_graph_with_namespaces() -> Graph:
    """Create and bind namespaces to a new graph"""
    g = Graph()
    bind_namespaces(g)
    return g

def generate_uri(prefix: str, seed: str) -> URIRef:
    """Centralized URI generation using UUID5"""
    uuid_val = uuid.uuid5(UUID_NAMESPACE, seed)
    return URIRef(KB[f"{prefix}-{uuid_val}"])

def add_individual_triples(g: Graph, entity: URIRef, label: str) -> None:
    """Add common individual triples (type and label)"""
    g.add((entity, RDF.type, OWL.NamedIndividual))
    g.add((entity, RDFS.label, Literal(label)))

def assign_statement_type(g: Graph, entity: URIRef, variable_data: Optional[Dict]) -> None:
    """Centralized logic for assigning statement types based on variable data"""
    if not variable_data:
        g.add((entity, RDF.type, PHB.NeomorphicStatement))
    elif variable_data.get("Variable comment"):
        g.add((entity, RDF.type, PHB.TransformationalComplexStatement))
    else:
        g.add((entity, RDF.type, PHB.TransformationalSimpleStatement))

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
    base.add((UBERON["0007023"], RDF.type, OWL.Class))
    base.add((UBERON["0003100"], RDF.type, OWL.Class))
    base.add((UBERON["0003101"], RDF.type, OWL.Class))

    base.add((UBERON["0007023"], RDFS.label, Literal("Adult Organism")))
    base.add((UBERON["0003100"], RDFS.label, Literal("Female Organism")))
    base.add((UBERON["0003101"], RDFS.label, Literal("Male Organism")))

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
shapes_graph = create_graph_with_namespaces()
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
    sp_g.add((sp_uri, RDF.type, TXR["0000006"]))  # the species is a TXR Species class

    # --- Species instance ---
    sp_instance = generate_uri("sp", sp_label.strip().lower())

    sp_g.add((sp_instance, RDF.type, sp_uri))  # species individual is an instance of the species class
    add_individual_triples(sp_g, sp_instance, sp_label)
    
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
    assign_statement_type(g, character, variable_data)
    g.add((character, RDF.type, OWL.NamedIndividual))

    # Print statement type for debugging
    if not variable_data:
        print(f"[Neomorphic] Char_ID: {data['Char_ID']}")
    
    elif variable_data.get("Variable comment"):
        print(f"[Transformational Complex] Char_ID: {data['Char_ID']}")
    else:
        print(f"[Transformational Simple] Char_ID: {data['Char_ID']}")

def handle_organism(
        g: Graph,
        org_data: Any,
        char_id: Optional[str] = None
    ) -> Optional[URIRef]:
    """
    Add RDF triples for the organism. Returns the organism instance URI.

    If a char_id is provided, the organism instance UUID is derived from
    (char_id + label), ensuring a distinct organism instance per Char_ID even
    when the organism label is identical across rows.
    """

    org_label = org_data.get("Label", "Unknown organism")
    org_uri_str = org_data.get("URI") or str(KB[org_label.replace(" ", "_")])
    org_uri = URIRef(org_uri_str)

    # Salt UUID with Char_ID (when available) to create per-character instances
    org_uuid_seed = (
        f"{char_id}::{org_label.strip().lower()}" if char_id else org_label.strip().lower()
    )
    org_instance = generate_uri("org", org_uuid_seed)

    g.add((org_uri, RDF.type, OWL.Class))
    g.add((org_uri, RDFS.label, Literal(org_label)))
    
    g.add((org_instance, RDF.type, org_uri))
    add_individual_triples(g, org_instance, org_label)

    return org_instance

def handle_locator(
    g: Graph,
    locator: Any,              
    parent_instance: URIRef,
    char_id: Optional[str] = None
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
    # Salt UUID with Char_ID to ensure per-character uniqueness
    seed_base = uri or label.strip().lower()
    seed = f"{char_id}::{seed_base}" if char_id else seed_base
    current_instance = generate_uri("loc", seed)

    if uri:
        class_uri = URIRef(uri)
        g.add((class_uri, RDFS.label, Literal(label)))
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((current_instance, RDF.type, class_uri))

    add_individual_triples(g, current_instance, label)

    # --- Chain anatomy ---
    g.add((parent_instance, BFO["0000051"], current_instance))  # previous → has_part → current

    return current_instance

def is_organism_label(label: Optional[str], target: str) -> bool:
    return bool(label) and label.strip().lower() == target

def is_adult_organism(org_data: Dict[str, Any]) -> bool:
    return is_organism_label(org_data.get("Label"), "adult organism")

def is_female_organism(org_data: Dict[str, Any]) -> bool:
    return is_organism_label(org_data.get("Label"), "female organism")

def is_male_organism(org_data: Dict[str, Any]) -> bool:
    return is_organism_label(org_data.get("Label"), "male organism")

def handle_organism_and_locators(
    g: Graph,
    data: Dict[str, Any],
    override_org: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[URIRef], List[URIRef]]:
    """
    Wrapper: handle organism + its locators.
    Returns a tuple:
        - organism_instance (or None)
        - list of locator instances in their chained order

    If override_org is provided, it will be used instead of data["Organism"].
    """

    # Decide which organism data to use
    organism_data = override_org if override_org is not None else (data.get("Organism") or {})

    # Create a per-Char_ID organism instance (salt UUID with Char_ID)
    organism_instance = handle_organism(
        g,
        organism_data,
        char_id=data.get("Char_ID")
    )
    previous_instance = organism_instance
    locators: List[URIRef] = []

    for locator in data.get("Locators", []) or []:
        # Ensure locator is a dict
        if isinstance(locator, str) or isinstance(locator, URIRef):
            locator = {"label": str(locator).split("/")[-1], "uri": str(locator)}
        elif not isinstance(locator, dict):
            print(f"[WARN] Unexpected locator type {type(locator)} for Char_ID {data.get('Char_ID')}")
            continue

        current_instance = handle_locator(g, locator, previous_instance, char_id=data.get("Char_ID"))
        if current_instance:
            locators.append(current_instance)
            previous_instance = current_instance  # chain continues

    return organism_instance, locators

def compute_default_organism_instance_uri_from_dataset(
        dataset: List[Dict[str, Any]]
    ) -> Optional[URIRef]:
    """
    Compute a canonical organism instance URI deterministically from the dataset.
    Uses the first row that has an Organism section and salts the UUID with that
    row's Char_ID to match the per-character organism instances created elsewhere.
    No triples are added here.
    """
    for row in dataset:
        org = row.get("Organism") or {}
        org_label = org.get("Label")
        char_id = row.get("Char_ID")
        if org_label and char_id:
            seed = f"{char_id}::{org_label.strip().lower()}"
            return generate_uri("org", seed)
    return None

def handle_variable_component(
    g: Graph,
    data: Dict[str, Any],
    char_id: Optional[str] = None
) -> Optional[URIRef]:
    """
    Add RDF triples for the 'Variable' section. Returns the variable instance URIRef or None.
    """
    var_data = data.get("Variable")
    if not var_data:
        return None

    

    # Salt UUID with Char_ID (when available) to create per-character instances
    var_label = var_data.get("Variable label", "Unnamed Variable")
    var_uri_str = var_data.get("Variable URI") or str(KB[var_label.replace(" ", "_")])
    var_uri = URIRef(var_uri_str)
    var_uuid_seed = (
        f"{char_id}::{var_label.strip().lower()}" if char_id else var_label.strip().lower()
    )
    var_instance_uri = generate_uri("var", var_uuid_seed)

    # var_instance_uri = generate_uri("var", f"{var_data.get('Variable URI', var_label.strip().lower())}")
    
    add_individual_triples(g, var_instance_uri, var_label)

    if var_data.get("Variable URI"):
        var_uri = URIRef(var_data["Variable URI"])
        g.add((var_instance_uri, RDF.type, var_uri))
        g.add((var_uri, RDFS.label, Literal(var_label)))

    if var_data.get("Variable comment"):
        g.add((var_instance_uri, RDFS.comment, Literal(var_data["Variable comment"])))

    return var_instance_uri

def handle_states(
    g: Graph,
    data: Dict[str, Any]
    # final_component: Optional[URIRef] = None
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
        state_node = generate_uri("sta", f"{data['Char_ID']}_{uri or label.lower()}")

        g.add((state_node, RDF.type, CDAO["0000045"]))  # CDAO State

        # Type assignment
        if uri:
            g.add((state_node, RDF.type, URIRef(uri)))
        add_individual_triples(g, state_node, label)

        # Link state to character
        state_map_for_char[state_index] = str(state_node)

    return state_map_for_char

def process_phenotype(
        g: Graph, 
        row: Dict[str, Any]
) -> Tuple[URIRef, Dict[int, str], Graph]:
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
          - state_map_for_char: Mapping of state indices to KB URIs.
          - sp_g: Species-specific RDFLib Graph (possibly empty).
    """
    char_id = row.get("Char_ID") or str(uuid.uuid4())
    char_label = row.get("CharacterLabel", f"Character {char_id}")

    # Character class definition
    char_uri = generate_uri("char", f"char_{char_id}")
    g.add((char_uri, RDF.type, CDAO["0000075"]))  # CDAO Character
    g.add((char_uri, RDFS.label, Literal(char_label)))

    # States: build state nodes and register allowed states per Character
    state_map_for_char = handle_states(g, row)

    # Attach Variable component at the character-level template (unique per Char_ID)
    var_instance_char = handle_variable_component(g, row, char_id=char_id)
    if var_instance_char:
        g.add((char_uri, PHB.has_variable_component, var_instance_char))
    
    # Catalog allowed states at the Character level only
    for idx, state_uri in state_map_for_char.items():

        g.add((char_uri, PHB.may_have_state, URIRef(state_uri)))
        print(f"[DEBUG] {char_label} (ID {char_id}) may_have_state -> {state_uri}")

    # Species Graph
    sp_g = create_graph_with_namespaces()
    sp_label = row.get("SpeciesLabel")
    species_id = row.get("SpeciesID")
    if sp_label and species_id:
        sp_uri = generate_uri("sp", sp_label)
        sp_g.add((sp_uri, RDF.type, PHB.Species))
        sp_g.add((sp_uri, RDFS.label, Literal(sp_label)))
        sp_g.add((sp_uri, DWC.parentNameUsageID, URIRef(f"https://www.gbif.org/species/{species_id}")))
        print(f"[DEBUG] Species graph for {sp_label} has {len(sp_g)} triples:")
        for s, p, o in sp_g:
            print(f"  {s} {p} {o}")

    return char_uri, state_map_for_char, sp_g

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
            "## --- KB#SP instances ---": lambda u: str(u).startswith(f"{KB_NS}sp-"),
            "## --- KB#PHE instances ---":  lambda u: str(u).startswith(f"{KB_NS}phe-"),
            "## --- KB#ORG instances ---": lambda u: str(u).startswith(f"{KB_NS}org-"),
            "## --- KB#LOC instances ---":  lambda u: str(u).startswith(f"{KB_NS}loc-"),
            "## --- KB#VAR instances ---":  lambda u: str(u).startswith(f"{KB_NS}var-"),
            "## --- KB#STA instances ---":    lambda u: str(u).startswith(f"{KB_NS}sta-"),
            "## --- KB#MX instances ---": lambda u: str(u).startswith(f"{KB_NS}mx-"),
            "## --- KB#CHAR instances ---": lambda u: str(u).startswith(f"{KB_NS}char-"),
            "## --- KB#TU instances ---":   lambda u: str(u).startswith(f"{KB_NS}tu-"),       
            "## --- KB#CELL instances ---": lambda u: str(u).startswith(f"{KB_NS}cell-"),
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
    combined_char_graph = create_graph_with_namespaces()
    for t in base_graph:
        combined_char_graph.add(t)

    character_graphs: Dict[str, Graph] = {}
    char_state_mapping: Dict[str, Dict[int, str]] = {}
    char_ids_in_order: List[str] = []

    for row in dataset:
        char_id = row["Char_ID"]
        char_ids_in_order.append(char_id)

        g_char = create_graph_with_namespaces()
        for t in base_graph:
            g_char.add(t)

        # Build phenotype and species graph
        char_uri, state_map_for_char, sp_g = process_phenotype(g_char, row)
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
    char_ids_in_order: List[str],
    char_state_mapping: Dict[str, Dict[int, str]]
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
    g = create_graph_with_namespaces()

    # Create matrix URI
    matrix_uri = URIRef(f"http://phenobees.org/kb#mx-{uuid.uuid4().hex[:8]}")
    g.add((matrix_uri, RDF.type, CDAO["0000056"]))  # CDAO Matrix
    g.add((matrix_uri, RDFS.label, Literal("matrix")))
    g.add((matrix_uri, DC.description, Literal("matrix description")))

    # TUs
    for taxon in nexus_matrix.taxon_namespace:
        tu_uri = generate_uri("tu", taxon.label)
        g.add((matrix_uri, CDAO["0000208"], tu_uri))  # CDAO has TU

    # Characters (columns) + phenotypes templates + species
    for char_index, char_id in enumerate(char_ids_in_order):
        # Pull row from dataset
        char_data = next((row for row in dataset if row["Char_ID"] == char_id), None)
        if not char_data:
            continue  # skip if no matching row

        # Process phenotype template (character-level; no state attachment here)
        char_uri, state_map, sp_g = process_phenotype(g, char_data)

        # Merge species triples
        if sp_g:
            g += sp_g

        # Link character into matrix
        g.add((matrix_uri, CDAO["0000142"], char_uri)) # CDAO has character

        # Cells (taxon x character)
        for taxon in nexus_matrix.taxon_namespace:
            # UUID-based cell
            cell_uri = generate_uri("cell", f"{taxon.label}_{char_index}")

            # UUID-based TU (for consistency)
            tu_uri = generate_uri("tu", taxon.label)

            g.add((cell_uri, RDF.type, CDAO["0000008"]))  # CDAO Cell
            g.add((cell_uri, CDAO["0000191"], tu_uri)) # CDAO belongs to TU
            g.add((cell_uri, CDAO["0000205"], char_uri)) # CDAO belongs to Character

            # Resolve the state for this specific cell
            cell_value = nexus_matrix[taxon][char_index]
            state_symbol = str(cell_value).strip() if cell_value is not None else ""

            chosen_state_node: Optional[URIRef] = None
            if state_symbol == '?':
                # Unknown state node (stable per character)
                unknown_node = generate_uri("unknown_state", f"{char_id}_unknown")
                add_individual_triples(g, unknown_node, "unknown")
                chosen_state_node = unknown_node
            elif state_symbol and state_symbol != '-':
                try:
                    state_index = int(state_symbol)
                    state_uri_str = char_state_mapping.get(char_id, {}).get(state_index)
                    if state_uri_str:
                        chosen_state_node = URIRef(state_uri_str)
                except ValueError:
                    pass  # unsupported polymorphic or non-integer state symbol

            # Create and link a per-cell phenotype statement that points to exactly one state
            per_pheno_seed = f"pheno-{char_id}::{str(taxon.label).strip().lower()}"
            per_pheno_uri = generate_uri("phe", per_pheno_seed)

            # Determine organism duplication needs
            org_data_in = char_data.get("Organism") or {}
            duplicate_for_sex = is_adult_organism(org_data_in) and not (is_female_organism(org_data_in) or is_male_organism(org_data_in))

            # Build list of phenotype variants to create
            pheno_variants: List[Tuple[URIRef, Optional[Dict[str, Any]]]] = []
            if duplicate_for_sex:
                # Create deterministic URIs for female/male variants
                phe_female = generate_uri("phe", per_pheno_seed + "::female")
                phe_male = generate_uri("phe", per_pheno_seed + "::male")
                pheno_variants.append((phe_female, {"Label": "female organism", "URI": str(UBERON["0003100"])}))
                pheno_variants.append((phe_male, {"Label": "male organism", "URI": str(UBERON["0003101"])}))
            else:
                pheno_variants.append((per_pheno_uri, None))

            for ph_uri, override_org in pheno_variants:
                # Minimal typing/label for the cell-specific phenotype
                add_individual_triples(g, ph_uri, f"Phenotype statement for {char_data.get('CharacterLabel', char_id)} in {taxon.label}")

                # Assign statement class based on Variable section
                variable_data = char_data.get("Variable")
                assign_statement_type(g, ph_uri, variable_data)

                # Attach the organism/locator/variable components (with optional override)
                org_instance, locator_instances = handle_organism_and_locators(g, char_data, override_org=override_org)
                if org_instance:
                    g.add((ph_uri, PHB.has_organismal_component, org_instance))
                for locator in locator_instances:
                    g.add((ph_uri, PHB.has_entity_component, locator))

                var_instance = handle_variable_component(g, char_data, char_id=char_id)
                if var_instance:
                    g.add((ph_uri, PHB.has_variable_component, var_instance))

                if var_instance and locator_instances:
                    g.add((locator_instances[-1], BFO["0000051"], var_instance))

                # Link exactly one state (if resolvable) to the cell phenotype instance
                if chosen_state_node is not None:
                    g.add((ph_uri, PHB.has_quality_component, chosen_state_node))

                    # Connect either Variable (preferred) or last Locator to the state via has_characteristic
                    if var_instance is not None:
                        g.add((var_instance, PHB.has_characteristic, chosen_state_node))
                    elif locator_instances:
                        last_locator = locator_instances[-1]
                        g.add((last_locator, PHB.has_characteristic, chosen_state_node))

                    # Cell also points to the state
                    g.add((cell_uri, CDAO["0000184"], chosen_state_node))  # Cell has_state

                # Link Cell → Phenotype (to the per-cell instance)
                g.add((cell_uri, PHB.refers_to_phenotype_statement, ph_uri))

    return g, matrix_uri

# ---------- TU processing & outputs ----------

def enrich_tu_graph(
    tu_graph: Graph,
    tu_uri: URIRef,
    taxon_label: str,
    species_info: Dict[str, Any],
    sp_uri: URIRef,
    sp_instance: URIRef,
    org_instance: URIRef,
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
        org_instance: Default organism instance URI.
        dir_combined: Output directory for TTL serialization.
    """
    # Format valid label
    valid_label = species_info.get("valid_species_name", taxon_label)
    parts = valid_label.split(" ", 2)
    binomial = f"{parts[0]} {parts[1]}" if len(parts) >= 2 else valid_label
    author = parts[2] if len(parts) == 3 else ""
    valid_label_html = f"<i>{binomial}</i> {author}".strip()

    # TU assertions
    add_individual_triples(tu_graph, tu_uri, valid_label_html)
    tu_graph.add((org_instance, RO["0003301"], tu_uri)) # RO has role in modelling
    tu_graph.add((tu_uri, RDF.type, CDAO["0000138"]))  # CDAO Taxon Unit
    tu_graph.add((tu_uri, IAO["0000219"], sp_instance)) # TU denotes species instance

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
    sp_graph = create_graph_with_namespaces()

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
    tu_uri = generate_uri("tu", f"{taxon_label.strip().lower()}")

    # Build a TU-specific graph
    tu_graph = create_graph_with_namespaces()

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
        # Unknown state: handled at matrix level by per-cell phenotype; skip here
        if state_symbol == '?':
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
            char_uri = generate_uri("char", f"char_{char_id}")
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
    org_instance = compute_default_organism_instance_uri_from_dataset(dataset)  # uses global 'dataset'

    # Enrich tu graph
    enrich_tu_graph(
        tu_graph,
        tu_uri,
        taxon_label,
        species_info,
        sp_uri,
        sp_instance,
        org_instance,
        dir_combined
    )

    return tu_graph  # return the TU graph for main to merge later

# ---------- MAIN Orchestration (single shared pipeline) ----------

def main():
    print("\n=== Building base graph and character graphs ===")
    base_graph = build_base_graph()

    # Combined report graph for all validations
    combined_report_graph = create_graph_with_namespaces()

    # Global species graph accumulator
    sp_g = create_graph_with_namespaces()

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
        char_ids_in_order=char_ids_in_order,
        char_state_mapping=char_state_mapping
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
    final_graph = create_graph_with_namespaces()

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

    # Render combined visualization
    combined_svg = os.path.join(DIR_GRAPHVIZ, "combined.svg")
    build_combined_visualization(final_graph, combined_svg)
    print(f"[OK] Combined graph visualization → {combined_svg}")

    # Serialize combined SHACL validation report graph
    validation_report_ttl = os.path.join(DIR_VALIDATION, "validation_report.ttl")
    try:
        combined_report_graph.serialize(destination=validation_report_ttl, format="turtle")
        print(f"[OK] SHACL combined validation report → {validation_report_ttl}")
    except Exception as e:
        print(f"[WARN] Failed to write SHACL validation report: {e}")

# ---------- Combined visualization ----------

def build_combined_visualization(
    g: Graph,
    output_path: str,
    tu_filter: Optional[List[str]] = None,
    prog: str = "sfdp",
    ranksep: float = 3.0,
    nodesep: float = 1.5,
    splines: str = "curved",
    cluster_by_tu: bool = False,
    max_label_len: int = 48,
    group_strategy: str = "buckets",  # initial | buckets | connected
    group_bucket_count: int = 6,
    add_spacers: bool = True
) -> None:
    """
    Build a single combined Graphviz visualization from the final merged RDF graph.

    Focus on core components and predicates already used in the pipeline:
      - Nodes: Matrix, TUs, Characters, Cells, Phenotypes, Species (concept + instance),
               Variables, Locators, Organisms, States.
      - Edges: key CDAO and PHB predicates wiring these components.
    """
    # Role detection helpers
    def is_uri(u):
        return isinstance(u, URIRef)

    # Resolve key predicate URIs for compact comparisons
    C_has_tu = CDAO["0000208"]       # Matrix has TU
    C_has_char = CDAO["0000142"]     # Matrix has Character
    C_cell = CDAO["0000008"]         # Cell class
    C_cell_belongs_tu = CDAO["0000191"]
    C_cell_belongs_char = CDAO["0000205"]
    C_cell_has_state = CDAO["0000184"]

    PH_has_org = PHB.has_organismal_component
    PH_has_ent = PHB.has_entity_component
    PH_has_var = PHB.has_variable_component
    PH_has_qual = PHB.has_quality_component
    PH_refers_pheno = PHB.refers_to_phenotype_statement
    PH_has_charac = PHB.has_characteristic

    IAO_denotes = IAO["0000219"]

    # Collect roles
    roles = {}
    labels = {}

    def label_for(u: URIRef) -> str:
        # Prefer rdfs:label if present
        lab = next((str(o) for o in g.objects(u, RDFS.label)), None)
        if lab:
            return lab
        # Fallback to suffix
        us = str(u)
        if '#' in us:
            return us.split('#')[-1]
        return us.rstrip('/').split('/')[-1]

    # Identify Matrix nodes
    matrix_nodes = set(s for s, p, o in g.triples((None, RDF.type, CDAO["0000056"])) )
    for m in matrix_nodes:
        roles[m] = "matrix"
        labels[m] = label_for(m)

    # Identify TU nodes via explicit typing or usage
    tu_nodes = set()
    for s, p, o in g:
        if p == C_has_tu and is_uri(o):
            tu_nodes.add(o)
        if p == C_cell_belongs_tu and is_uri(o):
            tu_nodes.add(o)
    # Also those typed as CDAO Taxon Unit
    for tu in g.subjects(RDF.type, CDAO["0000138"]):
        tu_nodes.add(tu)
    for tu in tu_nodes:
        roles[tu] = "tu"
        labels[tu] = label_for(tu)

    # Characters
    char_nodes = set()
    for s, p, o in g.triples((None, RDF.type, CDAO["0000075"])):
        char_nodes.add(s)
    for s, p, o in g:
        if p == C_has_char and is_uri(o):
            char_nodes.add(o)
        if p == C_cell_belongs_char and is_uri(o):
            char_nodes.add(o)
    for ch in char_nodes:
        roles[ch] = "character"
        labels[ch] = label_for(ch)

    # Cells
    cell_nodes = set(s for s, p, o in g.triples((None, RDF.type, C_cell)))
    for c in cell_nodes:
        roles[c] = "cell"
        labels[c] = label_for(c)

    # Phenotypes: look for nodes that are referenced by PHB.refers_to_phenotype_statement
    phe_nodes = set(o for s, p, o in g.triples((None, PH_refers_pheno, None)) if is_uri(o))
    for ph in phe_nodes:
        roles[ph] = "phenotype"
        labels[ph] = label_for(ph)

    # Species concept and instance: look for IAO:0000219 from TU → species instance
    sp_inst_nodes = set(o for s, p, o in g.triples((None, IAO_denotes, None)) if is_uri(o))
    for si in sp_inst_nodes:
        roles[si] = "species_instance"
        labels[si] = label_for(si)
    # Species concept nodes: class of species instance or TXR species class
    for si in sp_inst_nodes:
        for cls in g.objects(si, RDF.type):
            if is_uri(cls):
                roles.setdefault(cls, "species_concept")
                labels[cls] = label_for(cls)
    for sc in g.subjects(RDF.type, TXR["0000006"]):
        roles[sc] = "species_concept"
        labels[sc] = label_for(sc)

    # Variables, Locators, Organisms, States inferred from predicates
    var_nodes = set(o for s, p, o in g.triples((None, PH_has_var, None)) if is_uri(o))
    for v in var_nodes:
        roles[v] = "variable"
        labels[v] = label_for(v)

    loc_nodes = set(o for s, p, o in g.triples((None, PH_has_ent, None)) if is_uri(o))
    for l in loc_nodes:
        roles[l] = "locator"
        labels[l] = label_for(l)

    org_nodes = set(o for s, p, o in g.triples((None, PH_has_org, None)) if is_uri(o))
    for o_ in org_nodes:
        roles[o_] = "organism"
        labels[o_] = label_for(o_)

    state_nodes = set(o for s, p, o in g.triples((None, PH_has_qual, None)) if is_uri(o))
    state_nodes |= set(o for s, p, o in g.triples((None, C_cell_has_state, None)) if is_uri(o))
    for st in state_nodes:
        roles[st] = "state"
        labels[st] = label_for(st)

    # Build edges of interest
    edges = []  # (src, dst, label)
    def add_edge(s, d, label):
        edges.append((str(s), str(d), label))

    # Matrix edges
    for m in matrix_nodes:
        for tu in g.objects(m, C_has_tu):
            add_edge(m, tu, "hasTU")
        for ch in g.objects(m, C_has_char):
            add_edge(m, ch, "hasCharacter")

    # Cell edges
    for c in cell_nodes:
        for tu in g.objects(c, C_cell_belongs_tu):
            add_edge(c, tu, "belongsToTU")
        for ch in g.objects(c, C_cell_belongs_char):
            add_edge(c, ch, "belongsToCharacter")
        for st in g.objects(c, C_cell_has_state):
            add_edge(c, st, "hasState")
        for ph in g.objects(c, PH_refers_pheno):
            add_edge(c, ph, "refersToPhenotype")

    # Phenotype component edges
    for ph in phe_nodes:
        for o_ in g.objects(ph, PH_has_org):
            add_edge(ph, o_, "hasOrganism")
        for l in g.objects(ph, PH_has_ent):
            add_edge(ph, l, "hasEntity")
        for v in g.objects(ph, PH_has_var):
            add_edge(ph, v, "hasVariable")
        for st in g.objects(ph, PH_has_qual):
            add_edge(ph, st, "hasQuality")

    # Variable/Locator characteristic to State
    for v in var_nodes:
        for st in g.objects(v, PH_has_charac):
            add_edge(v, st, "hasCharacteristic")
    for l in loc_nodes:
        for st in g.objects(l, PH_has_charac):
            add_edge(l, st, "hasCharacteristic")

    # TU → species instance, species instance → species concept (rdf:type)
    for tu in tu_nodes:
        for si in g.objects(tu, IAO_denotes):
            add_edge(tu, si, "denotes")
            for sc in g.objects(si, RDF.type):
                if is_uri(sc):
                    add_edge(si, sc, "rdf:type")

    # Build node lists for visualization with role-based styling
    classes = []
    individuals = []
    # Color/shape maps
    style_map = {
        "matrix": ("box3d", "#d6d8db"),
        "tu": ("ellipse", "#ffe0b2"),
        "character": ("box", "#cce5ff"),
        "cell": ("diamond", "#f8d7da"),
        "phenotype": ("ellipse", "#d5f5e3"),
        "species_concept": ("hexagon", "#e2e3e5"),
        "species_instance": ("doublecircle", "#fff3cd"),
        "state": ("note", "#e2f0d9"),
        "variable": ("component", "#d1ecf1"),
        "locator": ("folder", "#e2e3e5"),
        "organism": ("house", "#f5e6ff"),
    }

    # Prepare Graphviz graph
    G = AGraph(directed=True, strict=False, rankdir="LR")
    G.node_attr.update(
        shape="ellipse", style="filled", fillcolor="#d5f5e3",
        margin="0.1,0.1", width="0.2", height="0.2",
        nodesep="1.0", ranksep="2.0", splines="true"
    )

    # Add nodes with styling
    for node, role in roles.items():
        n_id = str(node)
        label = labels.get(node, n_id.split('#')[-1])
        shape, color = style_map.get(role, ("ellipse", "#d5f5e3"))
        G.add_node(n_id, label=label, tooltip=n_id, shape=shape, fillcolor=color)
        # Partition nodes into classes/individuals arrays for potential subgraphs
        # Treat species_concept and character as classes; others as individuals
        if role in ("species_concept", "character"):
            classes.append(n_id)
        else:
            individuals.append(n_id)

    # Add edges
    for s, d, lbl in edges:
        G.add_edge(s, d, label=lbl)

    # Optional: group ranks
    if classes:
        G.add_subgraph(classes, rank='same')
    if individuals:
        G.add_subgraph(individuals, rank='same')

    # Layout and draw
    try:
        G.layout(prog="sfdp")
    except Exception:
        G.layout(prog="fdp")
    G.draw(output_path)


if __name__ == "__main__":
    main()