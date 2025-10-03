#!/bin/bash
# SPDX-FileCopyrightText: 2025 Thiago S. R. Silva, Diego S. Porto
# SPDX-License-Identifier: MIT

# Helper script to build and run the RDF Generator container
# Ensures input/output folders are correctly mounted

set -e  # Exit immediately if a command fails

# Build the Docker image
echo "Building Docker image..."
docker compose build

# Run the container
echo "Running RDF Generator..."
docker compose run --rm rdf-generator

echo "Done! Check the ./outputs folder for generated RDF graphs, PNGs, and reports."

