# RDF Generator

[![CI/CD Pipeline](https://github.com/tsrsilva/rdf-generator/actions/workflows/ci_cd.yaml/badge.svg)](https://github.com/tsrsilva/rdf-generator/actions)

A Python tool that generates **RDF phenotype graphs** from JSON and NEXUS matrices, applying **SHACL validation** and producing **graph visualizations** with Graphviz.

---

## Features

- Converts **NEXUS** and **JSON** phenotype data into **RDF** graphs  
- Validates RDF data using **SHACL**  
- Generates **Graphviz** visualizations (`.png`, `.svg`)  
- Organized input/output management  
- Tested via `pytest`  
- **Docker** and **CI/CD** pipeline with GitHub Actions

---

## Installation for development

Clone this repository and install dependencies locally:

```bash
git clone https://github.com/tsrsilva/rdf-generator.git
cd rdf-generator
pip install -e .
```

### Configuration
The pipeline is configured via a YAML file. Example:

```yaml
data_dir: "data"

input:
  json: "examples/minimal.json"
  nex: "examples/minimal.nex"
  species: "examples/species.json"
  shacl: "shapes/phenotype_shapes.ttl"

output:
  base_dir: "/data/outputs"
  ttl: "output_ttl"
  png: "output_png"
  validation: "validation_reports"
  combined: "combined_graphs"
  graphviz: "graphviz_images"
```

## Usage

For most collaborators, the easiest way to run RDF Generator is using Docker. This ensures consistent dependencies and paths, without requiring local Python setup.

### Option 1: Using Docker Compose (recommended)

```bash
docker compose build
docker compose run --rm rdf-generator
```

This will:

- Build the Docker image
- Run the container
- Automatically use the input data from ./data and configuration from ./configs/config.yaml
- Write outputs (RDF files, PNG visualizations, validation reports) to ./outputs on your host machine

No extra volume mounts or Docker commands are required.

### Option 2: Using the helper script

We provide a ```run.sh``` helper script to simplify running the container:

```bash
# Make the script executable (only needed once)
chmod +x run.sh

# Build and run the container
./run.sh
```

**Notes:**

- The script automatically handles building the image and mounting the correct directories.
- All generated outputs are saved to ./outputs on the host.
- Input data and configuration are automatically picked up from ./data and ./configs/config.yaml.

### Option 3: Building the Docker image manually

```bash
docker build -t rdf-generator .
docker run --rm rdf-generator
```

Using Option 1 or 2 is recommended to avoid manual volume mounts and ensure consistent input/output paths.

### Option 4: Running locally without Docker (not recommended)

If you prefer to run natively on Python:

```bash
python rdf_generator/main.py
```

Or inside Python:

```python
from rdf_generator import main

main.run()
```

- Ensure all dependencies from pyproject.toml are installed
- Make sure the data/ and configs/ directories are present in your project root
- Outputs will be saved according to the paths defined in configs/config.yaml

## Running tests

```bash
pytest tests/ --maxfail=1 --disable-warnings -v
```

## Project structure

```graphql
root/
├── rdf_generator/          # Main Python package with source code
├── data/                   # Input data: NEXUS, JSON, and SHACL files
├── outputs/                # Generated RDF, PNGs, validation reports, graphs
├── configs/                # Configuration files (YAML)
├── tests/                  # Pytest test suite
├── pyproject.toml          # Build configuration and dependencies
├── environment.yml         # Conda environment for development
├── docker-compose.yml      # Orchestrates Docker services for reproducible runs
├── Dockerfile              # Builds the RDF Generator container image
└── run.sh                  # Helper script to build/run the Docker container with proper mounts
```

## License

Licensed under the [MIT License](/LICENSES/MIT.txt)
© 2025 Thiago S. R. Silva, Diego S. Porto

## Funding

This tool was developed as part of the project “PhenoBees: a knowledgebase and integrative approach for studying the evolution of morphological traits in bees” funded by the Research Council of Finland (grant #362624).

## Authors
- [Thiago S. R. Silva](https://github.com/tsrsilva)
- [Diego S. Porto](https://github.com/diegosasso)