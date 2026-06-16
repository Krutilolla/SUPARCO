FROM continuumio/miniconda3

WORKDIR /app

# Copy environment file FIRST (important!)
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Ensure conda env is accessible
SHELL ["conda", "run", "-n", "suparco", "/bin/bash", "-c"]

# Copy rest of the backend code
COPY . .

EXPOSE 8000
