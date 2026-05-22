# Use a stable, lightweight Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system utilities and basic libraries required by MuJoCo/Gymnasium
RUN apt-get update && apt-get install -y --no-install-recommends \
    tree \
    nano \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch (CPU only) first to leverage Docker layer caching
# Using the explicit CPU index avoids downloading massive CUDA binaries
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Gymnasium (with MuJoCo) and other requested research libraries
# - pyyaml: required for configuration files
# - pytest: required for isolated testing
# - tqdm: standard for clean progress reporting in training loops
RUN pip install --no-cache-dir \
    "gymnasium[mujoco]" \
    numpy \
    matplotlib \
    pyyaml \
    pytest \
    tqdm

# Set the working directory
# This matches the mount point you will use in your docker run command
WORKDIR /sac-gymnasium

# Default command
CMD ["/bin/bash"]