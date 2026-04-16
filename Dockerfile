# SPDX-FileCopyrightText: 2025 Thiago S. R. Silva, Diego S. Porto
# SPDX-License-Identifier: MIT

# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy package metadata and source first so pip can install the local package.
COPY pyproject.toml /app/
COPY README.md /app/
COPY LICENSES /app/
COPY rdf_generator /app/rdf_generator

# Copy data folder (examples + shapes)
COPY data /app/data

# Install system dependencies for graphviz
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    pkg-config \
    graphviz \
    graphviz-dev \
    libc-dev \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies from pyproject.toml
RUN pip install --upgrade pip \
    && pip install .

# Copy the rest of the application
COPY configs /app/configs
COPY rdf_generator/main.py /app/

# Set entrypoint to run your main script
ENTRYPOINT ["python", "rdf_generator/main.py"]
