# Use the official Ubuntu 20.04 LTS as a base image
FROM ubuntu:20.04

# Avoid prompts from apt operations
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Set PATH for conda
ENV PATH /opt/conda/bin:$PATH

# Create a Conda environment
RUN conda create -n vh-env python=3.8 -y

# Activate the Conda environment in all subsequent RUN commands
SHELL ["conda", "run", "-n", "vh-env", "/bin/bash", "-c"]

# Install packages from requirements.txt using Conda from the conda-forge channel, excluding neo4j
# Assuming requirements.txt is modified to exclude neo4j==5.17.0=pypi_0
RUN conda install -c conda-forge --file requirements.txt -y

# Use pip to install neo4j, as it's available on PyPI
RUN pip install neo4j==5.17.0

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD pgrep -f slack_app.py || exit 1

# The code to run when container is started
ENTRYPOINT ["conda", "run", "-n", "vh-env", "python", "slack_app.py"]

