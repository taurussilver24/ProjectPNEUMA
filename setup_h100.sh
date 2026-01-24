#!/bin/bash
# PNEUMA H100 ENV SETUP (SFTP EDITION)
# Goal: Install dependencies & compile CUDA engine.
# Data/Models must be uploaded manually via SFTP.

set -e  # Exit on error

echo "🚀 INITIALIZING PNEUMA ENVIRONMENT..."

# 1. System Dependencies
echo "🛠️ Updating System..."
apt-get update && apt-get install -y git build-essential cmake python3-pip

# 2. Python Dependencies
echo "🔥 Installing Python Libs..."
pip install --upgrade pip
# Ensure pypdf is here since your scripts use it
pip install pandas openpyxl pypdf

# 3. THE CRITICAL STEP: Compile llama-cpp-python with CUDA support
# This binds Python to the H100. Without this, you get CPU speeds.
echo "⚡ Compiling CUDA Engine (Target: H100 SXM)..."
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

echo "✅ ENVIRONMENT READY."
echo "⚠️  REMINDER: You must now SFTP upload 'models/' and 'dataset/' folders"
echo "    into: /workspace/ProjectPNEUMA/"