#!/bin/bash

# Completely remove the old build directory and working venv.
# Important: reusing them can break the build, as described later.
deactivate
rm -rf build .venv-fc

# Create a venv that matches the FC runtime (Python 3.12), including bundled pip.
uv venv .venv-fc --python 3.12 --seed
source .venv-fc/bin/activate

# Copy only the required files into the build directory.
mkdir build
cp main.py build/
cp -r soccer_agent build/
cp -r static build/
rm -rf build/soccer_agent/__pycache__

# Workaround for the npx timeout issue.
npm install --prefix build mongodb-mcp-server
rm -rf build/node_modules/@oven


# Install dependencies directly under build/.
# Use the real pip inside the venv, not uv pip.
cd build
pip install -t . -r ../requirements.txt

# Create the zip archive.
# The key is to run this from the code package root directory.
zip -rq -y ../code.zip ./
cd ..
deactivate
