#!/bin/bash

# Create the main directory
mkdir -p got-bias-mitigation

# Navigate to the main directory
cd got-bias-mitigation

# Create the main files
touch got.py
touch requirements.txt
touch README.md

# Create the benchmarks directory and its contents
mkdir -p benchmarks/results
cd benchmarks
touch __init__.py
touch datasets.py
touch baselines.py
touch metrics.py
touch runner.py
cd ..

# Create the tests directory and its contents
mkdir -p tests
cd tests
touch test_got.py
touch test_metrics.py
cd ..

# Display the created structure
echo "Directory structure created successfully!"
echo
tree .