# SPDX-FileCopyrightText: 2025 Thiago S. R. Silva, Diego S. Porto
# SPDX-License-Identifier: MIT

import os
import json
import yaml
import uuid
from typing import Optional, Tuple, Dict, Any, List

import dendropy
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL
from pyshacl import validate
import itertools

# Global sequential counter for instance labels
PHENOTYPE_COUNTER = itertools.count(1)
ORGANISM_COUNTER = itertools.count(1)
LOCATOR_COUNTER = itertools.count(1)
VARIABLE_COUNTER = itertools.count(1)
QUALITY_COUNTER = itertools.count(1)
STATE_COUNTER = itertools.count(1)
MATRIX_COUNTER = itertools.count(1)
CHARACTER_COUNTER = itertools.count(1)
CELL_COUNTER = itertools.count(1)

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

# === SETUP ===
# Create output dirs if they don’t exist
for d in [DIR_VALIDATION, DIR_COMBINED]:
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
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")

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

# Robust Char_ID → integer parser for sorting; never raises.
# Returns a large default when parsing fails, ensuring deterministic ordering.
DEFAULT_CHAR_SORT_NUM = 10**9

def parse_char_num(char_id: Any) -> int:
    try:
        s = str(char_id).strip()
        if not s:
            return DEFAULT_CHAR_SORT_NUM
        # Remove a leading 'C' or 'c' if present
        if s[0] in ('C', 'c'):
            s = s[1:]
        # Try direct int
        return int(s)
    except Exception:
        # As a fallback, extract digits only
        try:
            digits = ''.join(ch for ch in str(char_id) if ch.isdigit())
            return int(digits) if digits else DEFAULT_CHAR_SORT_NUM
        except Exception:
            return DEFAULT_CHAR_SORT_NUM

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
        ("uberon", UBERON),
        ("xsd", XSD),
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

    # Minimal class declarations used across character graphs
    base.add((UBERON["0003100"], RDF.type, OWL.Class))
    base.add((UBERON["0003101"], RDF.type, OWL.Class))

    base.add((CDAO["0000008"], RDF.type, OWL.Class))  # CDAO Cell
    base.add((CDAO["0000045"], RDF.type, OWL.Class))  # CDAO State
    base.add((CDAO["0000056"], RDF.type, OWL.Class))  # CDAO Character matrix
    base.add((CDAO["0000075"], RDF.type, OWL.Class))  # CDAO Character
    base.add((CDAO["0000138"], RDF.type, OWL.Class))  # CDAO TU

    base.add((CDAO["0000008"], RDFS.label, Literal("standard cell")))
    base.add((CDAO["0000045"], RDFS.label, Literal("standard state")))
    base.add((CDAO["0000056"], RDFS.label, Literal("character state data matrix")))
    base.add((CDAO["0000075"], RDFS.label, Literal("standard character")))
    base.add((CDAO["0000138"], RDFS.label, Literal("TU")))

    base.add((PATO["0000001"], RDF.type, OWL.Class))  # PATO Quality

    base.add((PATO["0000001"], RDFS.label, Literal("quality")))

    base.add((PHB.NeomorphicStatement, RDF.type, OWL.Class))
    base.add((PHB.TransformationalSimpleStatement, RDF.type, OWL.Class))
    base.add((PHB.TransformationalComplexStatement, RDF.type, OWL.Class))

    base.add((PHB.NeomorphicStatement, RDFS.label, Literal("Neomorphic Statement")))
    base.add((PHB.TransformationalSimpleStatement, RDFS.label, Literal("Transformational Simple Statement")))
    base.add((PHB.TransformationalComplexStatement, RDFS.label, Literal("Transformational Complex Statement")))

    # Property declarations
    ## Object Properties
    base.add((PHB.has_organism_component, RDF.type, OWL.ObjectProperty))
    base.add((PHB.has_organism_component, RDFS.label, Literal("has organism component")))
    base.add((PHB.has_entity_component, RDF.type, OWL.ObjectProperty))
    base.add((PHB.has_entity_component, RDFS.label, Literal("has entity component")))
    base.add((PHB.has_variable_component, RDF.type, OWL.ObjectProperty))
    base.add((PHB.has_variable_component, RDFS.label, Literal("has variable component")))
    base.add((PHB.has_quality_component, RDF.type, OWL.ObjectProperty))
    base.add((PHB.has_quality_component, RDFS.label, Literal("has quality component")))
    base.add((PHB.refers_to_phenotype_statement, RDF.type, OWL.ObjectProperty))
    base.add((PHB.refers_to_phenotype_statement, RDFS.label, Literal("refers to phenotype statement")))
    base.add((CDAO["0000142"], RDF.type, OWL.ObjectProperty))
    base.add((CDAO["0000142"], RDFS.label, Literal("has_Character")))
    base.add((CDAO["0000184"], RDF.type, OWL.ObjectProperty))
    base.add((CDAO["0000184"], RDFS.label, Literal("has_State")))
    base.add((CDAO["0000191"], RDF.type, OWL.ObjectProperty))
    base.add((CDAO["0000191"], RDFS.label, Literal("belongs_to_TU")))
    base.add((CDAO["0000205"], RDF.type, OWL.ObjectProperty))
    base.add((CDAO["0000205"], RDFS.label, Literal("belongs_to_Character")))
    base.add((CDAO["0000208"], RDF.type, OWL.ObjectProperty))
    base.add((CDAO["0000208"], RDFS.label, Literal("has_TU")))
    base.add((BFO["0000051"], RDF.type, OWL.ObjectProperty))
    base.add((BFO["0000051"], RDFS.label, Literal("has part")))
    base.add((RO["0000053"], RDF.type, OWL.ObjectProperty)) # has_characteristic
    base.add((RO["0000053"], RDFS.label, Literal("has characteristic")))
    base.add((RO["0003301"], RDF.type, OWL.ObjectProperty)) # has_role_in_modelling
    base.add((RO["0003301"], RDFS.label, Literal("has role in modelling")))
    base.add((IAO["0000219"], RDF.type, OWL.ObjectProperty))
    base.add((IAO["0000219"], RDFS.label, Literal("denotes")))

    ## Datatype Properties
    base.add((DC.description, RDF.type, OWL.DatatypeProperty))
    base.add((DWC.taxonID, RDF.type, OWL.DatatypeProperty))
    base.add((DWC.parentNameUsageID, RDF.type, OWL.DatatypeProperty))
    base.add((KB.sortCharNum, RDF.type, OWL.DatatypeProperty))
    base.add((KB.sortCharNum, RDFS.label, Literal("sort character number")))
    base.add((KB.sortSpecies, RDF.type, OWL.DatatypeProperty))
    base.add((KB.sortSpecies, RDFS.label, Literal("sort species")))

    ## Annotation Properties
    base.add((RDFS.label, RDF.type, OWL.AnnotationProperty))
    base.add((RDFS.comment, RDF.type, OWL.AnnotationProperty))
    base.add((RDFS.seeAlso, RDF.type, OWL.AnnotationProperty))

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
    # Use valid_species_name for the label if available, otherwise fall back to sp_label
    concept_label = species_info.get("valid_species_name", sp_label)
    sp_g.add((sp_uri, RDFS.label, Literal(concept_label)))
    sp_g.add((sp_uri, RDF.type, TXR["0000006"]))  # the species is a TXR Species class

    # --- Species instance ---
    sp_instance = generate_uri("sp", sp_label.strip().lower())

    sp_g.add((sp_instance, RDF.type, sp_uri))  # species individual is an instance of the species class
    add_individual_triples(sp_g, sp_instance, sp_label)
    
    # If we have an external ID (GBIF), link it
    if species_info.get("ID"):
        gbif_uri = URIRef(f"https://www.gbif.org/species/{species_info['ID']}")
        sp_g.add((sp_instance, DWC.parentNameUsageID, Literal(f"GBIF:{species_info['ID']}")))
        sp_g.add((sp_instance, RDFS.seeAlso, gbif_uri))

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
        #char_id: Optional[str] = None,
        taxon_label: Optional[str] = None
    ) -> Optional[URIRef]:
    """
    Add RDF triples for the organism. Returns the organism instance URI.

    If a taxon name is provided, the organism instance UUID is derived from
    taxon_label, ensuring a distinct organism instance per species (taxon).

    """

    org_label = org_data.get("Label")
    org_uri_str = org_data.get("URI") or str(KB[org_label.replace(" ", "_")])
    org_uri = URIRef(org_uri_str)

    # Derive organism UUID solely from taxon_label (when available) to ensure per-species uniqueness
    org_label_norm = org_label.strip().lower() if org_label else "organism"
    taxon_norm = taxon_label.strip().lower() if taxon_label else None
    if taxon_norm:
        org_uuid_seed = f"{taxon_norm}::{org_label_norm}"
    else:
        org_uuid_seed = org_label_norm
    org_instance = generate_uri("org", org_uuid_seed)

    g.add((org_uri, RDF.type, OWL.Class))
    g.add((org_uri, RDFS.label, Literal(org_label)))
    
    g.add((org_instance, RDF.type, org_uri))
    g.add((org_instance, RDF.type, OWL.NamedIndividual))

    # Only add a sequential rdfs:label once to prevent multiple different labels
    if not any(g.objects(org_instance, RDFS.label)):
        seq_num = next(ORGANISM_COUNTER)
        # Use the original org label as prefix for the id (e.g., "female organism:id-3")
        prefix = org_label.strip() if org_label else "organism"
        add_individual_triples(g, org_instance, f"{prefix}:id-{seq_num}")

    # Preserve the human-friendly label as rdfs:comment if absent
    # if not any(g.objects(org_instance, RDFS.comment)):
        # g.add((org_instance, RDFS.comment, Literal(org_label)))

    return org_instance

def handle_locator(
    g: Graph,
    locator: Any,              
    parent_instance: URIRef,
    # char_id: Optional[str] = None,
    org_seed: Optional[str] = None
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
    # Salt UUID with Organism seed and Char_ID to ensure per-organism uniqueness
    seed_base = uri or label.strip().lower()
    if org_seed:
        seed = f"{org_seed}::{seed_base}"
    else:
        seed = seed_base
    current_instance = generate_uri("loc", seed)

    if uri:
        class_uri = URIRef(uri)
        g.add((class_uri, RDFS.label, Literal(label)))
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((current_instance, RDF.type, class_uri))

    # Add a single sequential rdfs:label only if none exists to avoid duplicates
    if not any(g.objects(current_instance, RDFS.label)):
        seq_num = next(LOCATOR_COUNTER)
        # Use the original locator label as prefix for the id (e.g., "labrum:id-5")
        prefix_loc = label.strip() if label else "locator"
        add_individual_triples(g, current_instance, f"{prefix_loc}:id-{seq_num}")
        
    # Keep the original locator label as a comment if absent
    # if not any(g.objects(current_instance, RDFS.comment)):
        # g.add((current_instance, RDFS.comment, Literal(label)))

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
    override_org: Optional[Dict[str, Any]] = None,
    taxon_label: Optional[str] = None
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

    # Create a per-species organism instance (salt UUID with taxon_label)
    organism_instance = handle_organism(
        g,
        organism_data,
        # char_id=data.get("Char_ID"),
        taxon_label=taxon_label
    )
    previous_instance = organism_instance
    locators: List[URIRef] = []

    org_seed = str(organism_instance) if organism_instance is not None else None

    for locator in data.get("Locators", []) or []:
        # Ensure locator is a dict
        if isinstance(locator, str) or isinstance(locator, URIRef):
            locator = {"label": str(locator).split("/")[-1], "uri": str(locator)}
        elif not isinstance(locator, dict):
            print(f"[WARN] Unexpected locator type {type(locator)} for Char_ID {data.get('Char_ID')}")
            continue

        if previous_instance is not None:
            current_instance = handle_locator(
                g,
                locator,
                previous_instance,
                org_seed=org_seed
            )
            if current_instance:
                locators.append(current_instance)
                previous_instance = current_instance  # chain continues
        else:
            print(f"[WARN] Skipping locator for Char_ID {data.get('Char_ID')} because previous_instance is None")

    return organism_instance, locators

def compute_default_organism_instance_uri_from_dataset(
        dataset: List[Dict[str, Any]]
    ) -> Optional[URIRef]:
    """
    Compute a canonical organism instance URI deterministically from the dataset.
    Uses the first row that has an Organism section and salts the UUID with that
    row's taxon_label to match the per-species organism instances created elsewhere.
    No triples are added here.
    """
    for row in dataset:
        org = row.get("Organism") or {}
        org_label = org.get("Label")
        taxon_label = row.get("SpeciesLabel")
        if org_label and taxon_label:
            seed = f"{taxon_label}::{org_label.strip().lower()}"
            return generate_uri("org", seed)
    return None

def handle_variable_component(
    g: Graph,
    data: Dict[str, Any],
    # char_id: Optional[str] = None,
    org_seed: Optional[str] = None
) -> Optional[URIRef]:
    """
    Add RDF triples for the 'Variable' section. Returns the variable instance URIRef or None.

    UUID seeding strategy:
      - Always include the variable label (normalized)
      - If available, include organism seed (per-species uniqueness)
      - If available, include the last locator of the locator chain (per-locator uniqueness)

    Note on scoping:
        Variable component in original character syntax scopes the quality
        to a specific entity. Current model treats variable as statement-level
        component, which may result in loss of scope information (i.e., the variable explains
        what entity?).
    
    Note on modeling: 
        Variables should be modeled as reified observation nodes:
        e.g.
            observation
                ├── observes_entity ──> lorum
                ├── has_variable ──> degree of fusion
                └── has_quality ──> fused
    """
    var_data = data.get("Variable")
    if not var_data:
        return None

    # Base identifiers
    var_label = var_data.get("Variable label", "Unnamed Variable")
    var_uri_str = var_data.get("Variable URI") or str(KB[var_label.replace(" ", "_")])
    var_uri = URIRef(var_uri_str)

    # Determine the last locator seed from the chain, preferring a URI if present
    last_loc_seed: Optional[str] = None
    try:
        locs = data.get("Locators") or []
        if isinstance(locs, list) and len(locs) > 0:
            last = locs[-1]
            if isinstance(last, dict):
                last_label = next((v for k, v in last.items() if "label" in k.lower()), None)
                last_uri = next((v for k, v in last.items() if "uri" in k.lower() and v), None)
                last_loc_seed = (last_uri or (last_label.strip().lower() if last_label else None))
            elif isinstance(last, (str, URIRef)):
                last_loc_seed = str(last)
    except Exception:
        # If anything goes wrong, just omit the locator from the seed
        last_loc_seed = None

    # Compose the UUID seed in a stable order
    seed_components: List[str] = []
    if org_seed:
        seed_components.append(str(org_seed))
    if last_loc_seed:
        seed_components.append(str(last_loc_seed))
    seed_components.append(var_label.strip().lower())

    var_uuid_seed = "::".join(seed_components)
    var_instance_uri = generate_uri("var", var_uuid_seed)

    # Add a single sequential label only if none exists
    if not any(g.objects(var_instance_uri, RDFS.label)):
        seq_num = next(VARIABLE_COUNTER)
        # Use the original variable label as prefix for the id (e.g., "shape:id-1")
        prefix_var = var_label.strip() if var_label else "variable"
        add_individual_triples(g, var_instance_uri, f"{prefix_var}:id-{seq_num}")
        
    # Preserve the original variable label for readability as comment
    # if not any(g.objects(var_instance_uri, RDFS.comment)):
        # g.add((var_instance_uri, RDFS.comment, Literal(var_label)))

    if var_data.get("Variable URI"):
        var_uri = URIRef(var_data["Variable URI"])
        g.add((var_uri, RDF.type, OWL.Class))
        g.add((var_uri, RDFS.label, Literal(var_label)))
        g.add((var_instance_uri, RDF.type, var_uri))

    if var_data.get("Variable comment"):
        g.add((var_instance_uri, RDFS.comment, Literal(var_data["Variable comment"])))

    return var_instance_uri

def handle_quality(
    g: Graph,
    data: Dict[str, Any]
) -> Dict[int, str]:
    """
    Add RDF triples for 'Qualities'. Returns a map of index -> quality node URI (str).
    Negations like "not X" are normalized into positive
    absence-style labels (e.g., "X absent"), avoiding owl:complementOf.
    """
    quality_map_for_char: Dict[int, str] = {}

    for quality_index, quality in enumerate(data.get("States", []) or []):
        label = next((v for k, v in quality.items() if 'label' in k.lower()), "unknown").strip()
        uri = next((v for k, v in quality.items() if 'uri' in k.lower() and v), None)

        norm_label = label.lower()

        # Detect negations
        if norm_label.startswith("not "):
            base_label = label[4:].strip()
            label = f"not {base_label}"

        # UUID for quality
        quality_node = generate_uri("qua", f"{data['Char_ID']}_{uri or label.lower()}")

        g.add((quality_node, RDF.type, PATO["0000001"]))  # PATO Quality

        # Type assignment
        if uri:
            g.add((URIRef(uri), RDF.type, OWL.Class))
            g.add((URIRef(uri), RDFS.label, Literal(label)))
            g.add((quality_node, RDF.type, URIRef(uri)))
        # Add a single sequential label only if none exists
        if not any(g.objects(quality_node, RDFS.label)):
            seq_num = next(QUALITY_COUNTER)
            add_individual_triples(g, quality_node, f"quality:id-{seq_num}")
        # Keep human-friendly label as comment if absent
        # if not any(g.objects(quality_node, RDFS.comment)):
            # g.add((quality_node, RDFS.comment, Literal(label)))

        # Link quality to character
        quality_map_for_char[quality_index] = str(quality_node)

    return quality_map_for_char

def handle_states(
    g: Graph,
    data: Dict[str, Any]
) -> Dict[int, str]:
    """
    Add RDF triples for 'States'. Returns a map of index -> state node URI (str).
    Negations like "not X" or "absent" are modeled as instances of an anonymous class that is
    the complement of a restriction: NOT (has_part some X).
    
    For negated states without their own URI:
        - "not X": look up URI or "X" in state_label_uri
        - "absent": look up URI of "present" in the same character
    """
    state_map_for_char: Dict[int, str] = {}

    for state_index, state in enumerate(data.get("States", []) or []):
        label = next((v for k, v in state.items() if 'label' in k.lower()), "unknown").strip()
        uri = next((v for k, v in state.items() if 'uri' in k.lower() and v), None)

        normalized_label = label.lower()
        # Detect negations: "not X" or "absent"
        is_negation = normalized_label.startswith("not ") or normalized_label == "absent"
        resolved_uri = uri  # will be used for complement restriction if negation
        base_label = None

        if is_negation:
            if normalized_label.startswith("not "):
                # "not X" pattern: extract base as "X"
                base_label = label[4:].strip()
                label = f"not {base_label}"

                # If this negation entry lacks a URI, look up the positive counterpart's URI
                if not uri:
                    base_label_normalized = base_label.strip().lower()
                    resolved_uri = state_label_to_uri.get(base_label_normalized)

            elif normalized_label == "absent":
                # "absent" pattern: use the last locator in the locator chain as the
                # target of the presence restriction (has_part some <locator>). If the
                # last locator supplies a URI, use it; otherwise create a skolemized
                # class for the locator label and use that.
                base_label = None
                resolved_uri = None
                locs = data.get("Locators") or []
                if isinstance(locs, list) and len(locs) > 0:
                    last = locs[-1]
                    last_label = next((v for k, v in last.items() if 'label' in k.lower()), None)
                    last_uri = next((v for k, v in last.items() if 'uri' in k.lower() and v), None)
                    if last_uri:
                        resolved_uri = last_uri
                        base_label = last_label if last_label else str(last_uri)
                    elif last_label:
                        # Create a skolemized class for the locator label so it can
                        # be used as the someValuesFrom target (must be a Class IRI).
                        seed = f"{data.get('Char_ID','') }::locator::{last_label.strip().lower()}"
                        loc_class = generate_uri("locclass", seed)
                        g.add((loc_class, RDF.type, OWL.Class))
                        g.add((loc_class, RDFS.label, Literal(last_label)))
                        resolved_uri = str(loc_class)
                        base_label = last_label
                # Fallback: if no locator-derived target was found, fall back to
                # the original strategy of using the "present" state's URI (if any)
                if not resolved_uri:
                    base_label = "present"
                    if not uri:
                        for other_state in data.get("States", []) or []:
                            other_label = next((v for k, v in other_state.items() if 'label' in k.lower()), "").strip().lower()
                            if other_label == "present":
                                other_uri = next((v for k, v in other_state.items() if 'uri' in k.lower() and v), None)
                                if other_uri:
                                    resolved_uri = other_uri
                                    break
                    if not resolved_uri:
                        resolved_uri = state_label_to_uri.get("present")
        
        # UUID for state
        state_node = generate_uri("sta", f"{data['Char_ID']}_{uri or label.lower()}")

        g.add((state_node, RDF.type, CDAO["0000045"]))  # CDAO State

        # Add human-readable label (sequential id label + original label as comment) only if absent
        if not any(g.objects(state_node, RDFS.label)):
            seq_num = next(STATE_COUNTER)
            prefix_sta = label.strip() if label else "state"
            add_individual_triples(g, state_node, f"{prefix_sta}:id-{seq_num}")
        # if not any(g.objects(state_node, RDFS.comment)):
            # g.add((state_node, RDFS.comment, Literal(label)))

        # If this is a negation with a resolved URI, model as:
        # state_node rdf:type [owl:complementOf [owl:Restriction ; owl:onProperty BFO:0000051 ; owl:someValuesFrom <uri>]]
        if is_negation and resolved_uri:
            # Create deterministic/skolemized nodes for the inner restriction and complement class
            # Use the resolved URI as the seed so the generated URIs are stable across runs/files
            seed = str(resolved_uri)
            inner_restriction = generate_uri("restr", seed)
            # Prefer to show the positive target label (base label or rdfs:label of the resolved URI)
            if base_label:
                display_target = base_label
            else:
                lbl = next((str(o) for o in g.objects(URIRef(resolved_uri), RDFS.label)), None)
                display_target = lbl if lbl else str(resolved_uri)

            # Use the positive target in the inner restriction label (explicit form: has_part some X)
            g.add((inner_restriction, RDFS.label, Literal(f"has_part some {display_target}")))
            g.add((inner_restriction, RDF.type, OWL.Restriction))
            g.add((inner_restriction, OWL.onProperty, BFO["0000051"]))  # has_part
            g.add((inner_restriction, OWL.someValuesFrom, URIRef(resolved_uri)))

            # Create the complement class: NOT (has_part some <uri>) using a stable URI
            complement_class = generate_uri("comp", seed)
            g.add((complement_class, OWL.complementOf, inner_restriction))
            g.add((complement_class, RDFS.label, Literal(f"NOT (has_part some {display_target})")))

            # Assert the state instance as an instance of the complement class
            g.add((state_node, RDF.type, complement_class))

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
          - quality_map_for_char: Mapping of quality indices to KB URIs.
          - state_map_for_char: Mapping of state indices to KB URIs.
          - sp_g: Species-specific RDFLib Graph (possibly empty).
    """
    char_id = row.get("Char_ID") or str(uuid.uuid4())
    char_label = row.get("CharacterLabel", f"Character {char_id}")

    # Character class definition
    char_uri = generate_uri("char", f"char_{char_id}")
    g.add((char_uri, RDF.type, CDAO["0000075"]))  # CDAO Character
    g.add((char_uri, RDFS.label, Literal(f"{char_label}")))
    g.add((char_uri, RDF.type, OWL.NamedIndividual))

    # States: build state nodes and register allowed states per Character
    state_map_for_char = handle_states(g, row)

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

def apply_matrix_label_priority(matrix_graph: Graph, target_graph: Graph) -> int:
    """Apply authoritative rdfs:label values from `matrix_graph` onto `target_graph`.

    For each subject present in `target_graph`, if `matrix_graph` contains
    one or more `rdfs:label` values for the same subject, replace any
    existing `rdfs:label` triples in `target_graph` with the first label
    found in `matrix_graph`.

    Returns the number of subjects changed.
    """
    changes = 0
    for subj in set(target_graph.subjects()):
        pref_labels = [str(l) for l in matrix_graph.objects(subj, RDFS.label)]
        if not pref_labels:
            continue
        preferred = pref_labels[0]
        existing = list(target_graph.objects(subj, RDFS.label))
        if not existing:
            target_graph.add((subj, RDFS.label, Literal(preferred)))
            changes += 1
        else:
            texts = [str(l) for l in existing]
            if not (len(texts) == 1 and texts[0] == preferred):
                for l in existing:
                    target_graph.remove((subj, RDFS.label, l))
                target_graph.add((subj, RDFS.label, Literal(preferred)))
                changes += 1
    return changes

def write_ttl_with_sections(graph: Graph, ttl_file: str) -> None:
    """Write TTL grouped into Classes, Individuals (grouped), Properties, and Other."""
    # Serialize once to get the prefix block
    ttl_full = graph.serialize(format="turtle", encoding="utf-8").decode("utf-8")
    prefix_block = ttl_full.split("\n\n", 1)[0]

    def _render_node(u):
            """
            Render a node using prefixed names when possible; fallback to <IRI>.
            Literals are rendered with .n3().
            """
            try:
                if isinstance(u, URIRef):
                    return graph.namespace_manager.normalizeUri(u)
                return u.n3()
            except Exception:
                return f"<{u}>" if isinstance(u, URIRef) else u.n3()  

    def _write_triple(f, s, p, o):
        s_txt = _render_node(s)
        p_txt = _render_node(p)
        o_txt = _render_node(o)
        f.write(f"{s_txt} {p_txt} {o_txt} .\n")

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
        f.write("### ===================== ### \n### ====== CLASSES ====== ###\n### ===================== ### \n\n")
        # === Classes ===

        # Preferred predicate ordering within class blocks
        org_preferred_preds = [RDFS.label, RDF.type]

        for s in sorted(class_nodes, key=lambda x: str(x)):
            # Collect predicate-object pairs for this class subject
            pos = list(graph.predicate_objects(s))
            if not pos:
                continue

            # Sort by preferred order first, then by predicate IRI and object
            def pred_rank(p):
                try:
                    return org_preferred_preds.index(p)
                except ValueError:
                    return len(org_preferred_preds)

            pos.sort(key=lambda po: (pred_rank(po[0]), str(po[0]), str(po[1])))

            # Per-class heading
            f.write(f"### {_render_node(s)}\n")

            # Emit compact Turtle block with semicolons and final period
            subj_txt = _render_node(s)
            for idx, (p, o) in enumerate(pos):
                pred_txt = _render_node(p)
                obj_txt = _render_node(o) if isinstance(o, URIRef) else o.n3()
                is_last = (idx == len(pos) - 1)
                if idx == 0:
                    line = f"{subj_txt} {pred_txt} {obj_txt} {' .' if is_last else ' ;'}"
                else:
                    line = f"  {pred_txt} {obj_txt} {' .' if is_last else ' ;'}"
                # Normalize spaces before terminators
                line = line.replace('  .', ' .').replace('  ;', ' ;')
                f.write(line + "\n")
            f.write("\n")

        # === Individuals (grouped by KB prefixes) ===
        f.write("### ===================== ### \n### ==== INDIVIDUALS ==== ###\n### ===================== ### \n\n")
        # === Individuals ===

        # Group by prefix buckets
        buckets = {
            "## --- Species instances --- ##": lambda u: str(u).startswith(f"{KB_NS}sp-"),
            "## --- Phenotype instances --- ##":  lambda u: str(u).startswith(f"{KB_NS}phe-"),
            "## --- Organism instances --- ##": lambda u: str(u).startswith(f"{KB_NS}org-"),
            "## --- Locator instances --- ##":  lambda u: str(u).startswith(f"{KB_NS}loc-"),
            "## --- Variable instances --- ##":  lambda u: str(u).startswith(f"{KB_NS}var-"),
            "## --- Quality instances --- ##":   lambda u: str(u).startswith(f"{KB_NS}qua-"),
            "## --- State instances --- ##":    lambda u: str(u).startswith(f"{KB_NS}sta-"),
            "## --- Matrix instances --- ##": lambda u: str(u).startswith(f"{KB_NS}mx-"),
            "## --- Character instances --- ##": lambda u: str(u).startswith(f"{KB_NS}char-"),
            "## --- TU instances --- ##":   lambda u: str(u).startswith(f"{KB_NS}tu-"),       
            "## --- Cell instances --- ##": lambda u: str(u).startswith(f"{KB_NS}cell-"),
            "## --- Other Individuals --- ##":  lambda u: True,  # fallback
        }

        idv_preferred_preds = [
            RDFS.label, 
            RDF.type, 
            DWC.parentNameUsageID, 
            RDFS.seeAlso,
            PHB.has_organism_component,
            PHB.has_entity_component,
            PHB.has_variable_component,
            PHB.has_quality_component,
            PHB.may_have_state,
            PHB.refers_to_phenotype_statement,
            BFO["0000051"],
            RO["0000053"],
            RO["0003301"],
            IAO["0000219"],
            CDAO["0000184"],
            CDAO["0000191"],
            CDAO["0000205"],
            CDAO["0000142"],
            CDAO["0000208"]
        ]

        remaining = set(individual_nodes)
        for header, pred in buckets.items():
            bucket_nodes = [u for u in remaining if pred(u)]
            if not bucket_nodes:
                continue

            # Write bucket header
            f.write(header + "\n\n")

            # Prefer metadata-based sorting for Phenotype instances to avoid brittle label parsing
            def phenotype_sort_key(u: URIRef):
                # Defaults push non-matching to the end deterministically
                default = (10**9, "")
                try:
                    # Read kb:sortCharNum (integer) and kb:sortSpecies (string) if present
                    sort_char = next((o for o in graph.objects(u, KB.sortCharNum)), None)
                    sort_species = next((o for o in graph.objects(u, KB.sortSpecies)), None)
                    if sort_char is not None and sort_species is not None:
                        try:
                            char_num = int(str(sort_char))
                        except (ValueError, TypeError):
                            # If literal has datatype, try toPython() conversion
                            try:
                                char_num = int(sort_char.toPython())
                            except Exception:
                                return default
                        species = str(sort_species)
                        return (char_num, species)
                except Exception:
                    pass
                return default

            if "Phenotype" in header:
                sorted_nodes = sorted(bucket_nodes, key=lambda u: (phenotype_sort_key(u), str(u)))
            else:
                sorted_nodes = sorted(bucket_nodes, key=lambda x: str(x))

            for s in sorted_nodes:
                pos = list(graph.predicate_objects(s))
                if not pos:
                    continue

                # Sort by preferred order first, then by predicate IRI and object
                def pred_rank(p):
                    try:
                        return idv_preferred_preds.index(p)
                    except ValueError:
                        return len(idv_preferred_preds)

                pos.sort(key=lambda po: (pred_rank(po[0]), str(po[0]), str(po[1])))

                f.write(f"### {_render_node(s)}\n")

                subj_txt = _render_node(s)
                for idx, (p, o) in enumerate(pos):
                    pred_txt = _render_node(p)
                    obj_txt = _render_node(o)
                    is_last = (idx == len(pos) - 1)
                    if idx == 0:
                        line = f"{subj_txt} {pred_txt} {obj_txt} {' .' if is_last else ' ;'}"
                    else:
                        line = f"  {pred_txt} {obj_txt} {' .' if is_last else ' ;'}"
                    line = line.replace('  .', ' .').replace('  ;', ' ;')
                    f.write(line + "\n")
                f.write("\n")
            # Remove written nodes so they don't show again
            remaining -= set(bucket_nodes)

        # === Properties ===
        f.write("### ==============================\n### Properties\n### ==============================\n\n")

        prp_preferred_preds = [RDFS.label, RDF.type]

        def write_prop_section(title: str, nodes: set):
            if not nodes:
                return

            # Sort by preferred order first, then by predicate IRI and object
            def pred_rank(p):
                try:
                    return prp_preferred_preds.index(p)
                except ValueError:
                    return len(prp_preferred_preds)

            # Write section title
            f.write(title + "\n\n")
            for s in sorted(nodes, key=lambda x: str(x)):
                pos = list(graph.predicate_objects(s))
                if not pos:
                    continue

                subj_txt = _render_node(s)
                for idx, (p, o) in enumerate(pos):
                    pred_txt = _render_node(p)
                    obj_txt = _render_node(o)
                    is_last = (idx == len(pos) - 1)
                    if idx == 0:
                        line = f"{subj_txt} {pred_txt} {obj_txt} {' .' if is_last else ' ;'}"
                    else:
                        line = f"  {pred_txt} {obj_txt} {' .' if is_last else ' ;'}"
                    line = line.replace('  .', ' .').replace('  ;', ' ;')
                    f.write(line + "\n")
                f.write("\n")

        write_prop_section("## --- ObjectProperties ---", object_properties)
        write_prop_section("## --- DatatypeProperties ---", data_properties)
        write_prop_section("## --- AnnotationProperties ---", annot_properties)

        # === Other Triples ===
        f.write("### ==============================\n### Other Triples\n### ==============================\n\n")
        from collections import defaultdict

        other_by_subject = defaultdict(list)
        for s, p, o in graph:
            if s in excluded_subjects:
                continue
            other_by_subject[s].append((p, o))

        for s in sorted(other_by_subject.keys(), key=lambda x: str(x)):
            pos = sorted(other_by_subject[s], key=lambda po: (str(po[0]), str(po[1])))
            f.write(f"### {_render_node(s)}\n")
            subj_txt = _render_node(s)
            for idx, (p, o) in enumerate(pos):
                pred_txt = _render_node(p)
                obj_txt = _render_node(o)
                is_last = (idx == len(pos) - 1)
                if idx == 0:
                    line = f"{subj_txt} {pred_txt} {obj_txt} {' .' if is_last else ' ;'}"
                else:
                    line = f"  {pred_txt} {obj_txt} {' .' if is_last else ' ;'}"
                line = line.replace('  .', ' .').replace('  ;', ' ;')
                f.write(line + "\n")
            f.write("\n")

def prune_unreferenced_prototypes(g: Graph) -> Dict[str, int]:
    """
    Remove unreferenced prototype individuals:
      - Qualities: kb:qua-* that are not objects of phb:has_quality_component
      - Organisms: kb:org-* that are not objects of phb:has_organism_component
    Returns counts removed by kind.
    """
    KB_NS = str(KB)
    removed = {"qualities": 0, "organisms": 0, "total": 0}

    referenced_qualities = {o for _, _, o in g.triples((None, PHB.has_quality_component, None)) if isinstance(o, URIRef)}
    referenced_organisms = {o for _, _, o in g.triples((None, PHB.has_organism_component, None)) if isinstance(o, URIRef)}

    candidates = list(set(g.subjects(RDF.type, OWL.NamedIndividual)))
    to_remove: List[Tuple[str, URIRef]] = []

    for s in candidates:
        if not isinstance(s, URIRef):
            continue
        su = str(s)
        if su.startswith(f"{KB_NS}qua-"):
            if s not in referenced_qualities:
                to_remove.append(("qualities", s))
        elif su.startswith(f"{KB_NS}org-"):
            if s not in referenced_organisms:
                to_remove.append(("organisms", s))

    for kind, s in to_remove:
        # Remove outgoing triples
        for p, o in list(g.predicate_objects(s)):
            g.remove((s, p, o))
        # Remove incoming triples
        for subj, pred in list(g.subject_predicates(s)):
            g.remove((subj, pred, s))
        removed[kind] += 1
        removed["total"] += 1

    print(f"[PRUNE] Removed {removed['qualities']} unreferenced quality prototypes and {removed['organisms']} unreferenced organism prototypes (total {removed['total']}).")
    return removed

# Visualization disabled: visualize_graph removed

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
    char_quality_mapping: Dict[str, Dict[int, str]],
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

    # Create matrix URI using a stable prefix
    mx_label = row.get("MatrixLabel")
    matrix_uri = generate_uri("mx", mx_label or "default_matrix")
    g.add((matrix_uri, RDF.type, CDAO["0000056"]))  # CDAO Matrix/CharacterStateDataMatrix
    # Add sequential label only if not set already
    if not any(g.objects(matrix_uri, RDFS.label)):
        seq_num = next(MATRIX_COUNTER)
        g.add((matrix_uri, RDFS.label, Literal(f"matrix:id-{seq_num}")))
    g.add((matrix_uri, DC.description, Literal("matrix description")))
    g.add((matrix_uri, RDF.type, OWL.NamedIndividual))

    # Characters (columns) + phenotypes templates + species
    for char_index, char_id in enumerate(char_ids_in_order):
        # Pull row from dataset
        char_data = next((row for row in dataset if row["Char_ID"] == char_id), None)
        if not char_data:
            continue  # skip if no matching row

        # Precompute quality nodes for this character once and register mapping
        quality_map_for_char = handle_quality(g, char_data)
        char_quality_mapping[char_id] = quality_map_for_char

        # Process phenotype template (character-level; no state attachment here)
        char_uri, state_map, sp_g = process_phenotype(g, char_data)

        # Merge species triples
        if sp_g:
            g += sp_g

        # Link character into matrix
        g.add((matrix_uri, CDAO["0000142"], char_uri))  # CDAO has character

        # Cells (taxon x character)
        for taxon in nexus_matrix.taxon_namespace:
            # UUID-based cell
            cell_uri = generate_uri("cell", f"{taxon.label}_{char_index}")

            # Sequential label for cell instances (only if not already present)
            if not any(g.objects(cell_uri, RDFS.label)):
                seq_num = next(CELL_COUNTER)
                g.add((cell_uri, RDFS.label, Literal(f"cell:id-{seq_num}")))
            if not any(g.objects(cell_uri, DC.description)):
                g.add((cell_uri, DC.description, Literal(f"Cell for taxon {taxon.label}, character {char_id}")))
            g.add((cell_uri, RDF.type, OWL.NamedIndividual))
            g.add((cell_uri, RDF.type, CDAO["0000008"]))  # CDAO Cell
            # CDAO belongs to TU: added after minting organism-specific TU later in the loop
            g.add((cell_uri, CDAO["0000205"], char_uri))  # CDAO belongs to Character

            # Resolve the state for this specific cell
            cell_value = nexus_matrix[taxon][char_index]
            state_symbol = str(cell_value).strip() if cell_value is not None else ""

            # Guard: only proceed with phenotype creation if state_symbol is a valid integer
            try:
                state_index = int(state_symbol)
            except (TypeError, ValueError):
                # Skip phenotype creation and linking for non-integer states (e.g., '-', '?', polymorphic)
                continue

            # Resolve chosen_state_node for valid integer state
            chosen_state_node: Optional[URIRef] = None
            state_uri_str = char_state_mapping.get(char_id, {}).get(state_index)
            if state_uri_str:
                chosen_state_node = URIRef(state_uri_str)

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
                g.add((ph_uri, KB.sortCharNum, Literal(parse_char_num(char_id), datatype=XSD.integer)))
                g.add((ph_uri, KB.sortSpecies, Literal(taxon.label)))
                g.add((ph_uri, DC.description, Literal(f"Phenotype statement for {char_data.get('CharacterLabel', char_id)} in {taxon.label}")))
                # Sequential, human-friendly label for phenotype instances
                seq_num = next(PHENOTYPE_COUNTER)
                add_individual_triples(g, ph_uri, f"phenotype:id-{seq_num}")

                # Assign statement class based on Variable section
                variable_data = char_data.get("Variable")
                assign_statement_type(g, ph_uri, variable_data)

                # Attach the organism/locator/variable components (with optional override)
                org_instance, locator_instances = handle_organism_and_locators(
                    g,
                    char_data,
                    override_org=override_org,
                    taxon_label=str(taxon.label)
                )

                # Mint a TU unique to species and connect organism, cell and matrix to it
                tu_seed = f"{str(taxon.label).strip().lower()}"
                tu_uri = generate_uri("tu", tu_seed)
                g.add((tu_uri, RDF.type, OWL.NamedIndividual))
                g.add((tu_uri, RDF.type, CDAO["0000138"]))

                if org_instance is not None:
                    g.add((org_instance, RO["0003301"], tu_uri))
                g.add((cell_uri, CDAO["0000191"], tu_uri))
                g.add((matrix_uri, CDAO["0000208"], tu_uri))

                if org_instance:
                    g.add((ph_uri, PHB.has_organism_component, org_instance))
                for locator in locator_instances:
                    g.add((ph_uri, PHB.has_entity_component, locator))

                var_instance = handle_variable_component(
                    g,
                    char_data,
                    org_seed=(str(org_instance) if org_instance is not None else None)
                )
                if var_instance:
                    g.add((ph_uri, PHB.has_variable_component, var_instance))

                # Unify quality resolution: use the same index as state_symbol, or '?' → unknown quality
                chosen_quality_node: Optional[URIRef] = None
                # Use the same numeric index for the quality node as the state index
                q_uri_str = quality_map_for_char.get(state_index)
                if q_uri_str:
                    chosen_quality_node = URIRef(q_uri_str)

                if chosen_quality_node is not None:
                    # If this is a known quality (not '?'), mint a per-organism quality node
                    if state_symbol != '?' and org_instance is not None:
                        base_q = chosen_quality_node
                        per_org_q_seed = f"{str(org_instance)}::{char_id}::{str(base_q)}"
                        per_org_quality = generate_uri("qua", per_org_q_seed)
                        # Copy label and types from base quality node
                        base_label = next((str(o) for o in g.objects(base_q, RDFS.label)), None)
                        add_individual_triples(g, per_org_quality, base_label or "quality")
                        g.add((per_org_quality, RDF.type, PATO["0000001"]))
                        for t in g.objects(base_q, RDF.type):
                            if t != OWL.NamedIndividual:
                                g.add((per_org_quality, RDF.type, t))
                        chosen_quality_node = per_org_quality

                    # Connect only the last Locator to the quality via has_characteristic (RO:0000053)
                    if locator_instances:
                        last_locator = locator_instances[-1]
                        g.add((last_locator, RO["0000053"], chosen_quality_node))
                    # Still assert that the phenotype has this quality component
                    g.add((ph_uri, PHB.has_quality_component, chosen_quality_node))

                # Link exactly one state (if resolvable) to the cell phenotype instance
                if chosen_state_node is not None:
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

    # TU assertions: assign TU name as taxon_label and keep human label as comment
    add_individual_triples(tu_graph, tu_uri, f"{taxon_label}")
    tu_graph.add((tu_uri, RDFS.comment, Literal(valid_label_html)))
    # Only assert org_instance triples if we have a valid URIRef
    if org_instance is not None:
        tu_graph.add((org_instance, RDF.type, OWL.NamedIndividual))
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
    # dir_graphviz: str
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

    # Fallback: synthesize a per-taxon organism instance if not derivable from dataset
    if org_instance is None:
        # Use a generic organism label; UUID seed includes taxon to remain deterministic per TU
        fallback_seed = f"{taxon_label.strip().lower()}::organism"
        org_instance = generate_uri("org", fallback_seed)

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

    char_quality_mapping: Dict[str, Dict[int, str]] = {}

    print("\n=== Building CDAO matrix graph (shared pipeline) ===")
    matrix_graph, matrix_uri = build_cdao_matrix(
    nexus_matrix=char_matrix,
    dataset=dataset,
    char_ids_in_order=char_ids_in_order,
    char_quality_mapping=char_quality_mapping,
    char_state_mapping=char_state_mapping
    )
    
    # Validate Matrix graph against SHACL shapes
    validate_graph_and_record(
    entity_id="CDAO Matrix",
    g=matrix_graph,
    shapes=shapes_graph,
    combined_report_graph=combined_report_graph,
    validation_dir=DIR_VALIDATION
    )
    
    print("\n=== Processing each taxon (TU graphs) ===")
    tu_graphs = []  # collect all TU graphs
    for taxon in char_matrix.taxon_namespace:
        g_tu = process_taxon(
            taxon=taxon,
            sp_g=sp_g,
            species_data=species_data,
            matrix_graph=matrix_graph,
            character_graphs=character_graphs,
            char_state_mapping=char_state_mapping,
            char_ids_in_order=char_ids_in_order,
            shapes=shapes_graph,
            combined_report_graph=combined_report_graph,
            validation_dir=DIR_VALIDATION,
            dir_combined=DIR_COMBINED,
            # dir_graphviz=DIR_GRAPHVIZ
        )
        tu_graphs.append(g_tu)

    # Validate the combined species accumulator graph
    validate_graph_and_record(
        entity_id="Species Combined",
        g=sp_g,
        shapes=shapes_graph,
        combined_report_graph=combined_report_graph,
        validation_dir=DIR_VALIDATION
    )

    # Write individual components (optional)
    # Prune unreferenced prototypes at the matrix level to avoid dangling individuals
    prune_unreferenced_prototypes(matrix_graph)
    matrix_ttl = os.path.join(DIR_COMBINED, "matrix.ttl")
    write_ttl_with_sections(matrix_graph, matrix_ttl)

    chars_ttl = os.path.join(DIR_COMBINED, "characters_combined.ttl")
    # Ensure matrix labels are authoritative for character graph
    try:
        changes = apply_matrix_label_priority(matrix_graph, combined_char_graph)
        if changes:
            print(f"[INFO] Applied matrix label priority to character graph ({changes} subjects updated)")
    except Exception as e:
        print(f"[WARN] Failed to apply matrix label priority to character graph: {e}")
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

    # Validate final merged graph prior to writing TTL
    validate_graph_and_record(
        entity_id="Final Combined Graph",
        g=final_graph,
        shapes=shapes_graph,
        combined_report_graph=combined_report_graph,
        validation_dir=DIR_VALIDATION
    )

    # Final pruning on the merged graph
    prune_unreferenced_prototypes(final_graph)

    merged_ttl = os.path.join(DIR_COMBINED, "all_combined.ttl")
    # Ensure matrix labels are authoritative in the final merged graph as well
    try:
        changes = apply_matrix_label_priority(matrix_graph, final_graph)
        if changes:
            print(f"[INFO] Applied matrix label priority to merged graph ({changes} subjects updated)")
    except Exception as e:
        print(f"[WARN] Failed to apply matrix label priority to merged graph: {e}")
    write_ttl_with_sections(final_graph, merged_ttl)
    print(f"[OK] Full combined TTL → {merged_ttl}")

    # Serialize combined SHACL validation report graph
    validation_report_ttl = os.path.join(DIR_VALIDATION, "validation_report.ttl")
    try:
        combined_report_graph.serialize(destination=validation_report_ttl, format="turtle")
        print(f"[OK] SHACL combined validation report → {validation_report_ttl}")
    except Exception as e:
        print(f"[WARN] Failed to write SHACL validation report: {e}")

# Combined visualization disabled: build_combined_visualization removed


if __name__ == "__main__":
    main()