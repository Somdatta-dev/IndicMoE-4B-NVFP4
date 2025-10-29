#!/bin/bash
# Setup data directories for IndicMoE-4B training

set -e

echo "=========================================="
echo "Setting up IndicMoE-4B data directories"
echo "=========================================="

# Create directory structure
echo "Creating directory structure..."

mkdir -p /workspace/data/raw/phase1_pretraining
mkdir -p /workspace/data/raw/phase2_instruction
mkdir -p /workspace/data/raw/phase3_function_calling

mkdir -p /workspace/data/processed/phase1_pretraining
mkdir -p /workspace/data/processed/phase2_instruction
mkdir -p /workspace/data/processed/phase3_function_calling
mkdir -p /workspace/data/processed/tokenizer

mkdir -p /workspace/data/cache

mkdir -p /workspace/logs
mkdir -p /workspace/runs
mkdir -p /workspace/checkpoints
mkdir -p /workspace/configs

echo "✓ Directory structure created"

# Set permissions
echo "Setting permissions..."
chmod -R 755 /workspace/data
chmod -R 755 /workspace/logs
chmod -R 755 /workspace/runs
chmod -R 755 /workspace/checkpoints

echo "✓ Permissions set"

# Display structure
echo ""
echo "Directory structure:"
tree -L 3 /workspace/data || find /workspace/data -type d | sed 's|[^/]*/| |g'

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="

