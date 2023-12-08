# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# Set non-interactive shell to avoid prompts during build
ARG DEBIAN_FRONTEND=noninteractive

# Remove any third-party apt sources to avoid issues with expiring keys
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    unzip \
 && apt-get clean && \
    rm -rf /var/lib/apt/lists/

# Create and set the working directory
WORKDIR /app
ENV HOME=/home

# Set up Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=$HOME/miniconda/bin:$PATH

COPY environment.yml /app/environment.yml

RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ~/miniconda && \
    rm ~/miniconda.sh

RUN conda env create --file=environment.yml && \
    rm /app/environment.yml && \
    conda clean -ya

# Configure shell to activate Conda environment by default
RUN echo "source activate rcp" > ~/.bashrc
ENV PATH /opt/conda/env/rcp/bin:$PATH

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "rcp", "/bin/bash", "-c"]