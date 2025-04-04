FROM ubuntu:24.04
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up a virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install dependencies in the virtual environment
COPY docker-requirements.txt .
RUN pip install --no-cache-dir -r docker-requirements.txt

# Set entrypoint (modify as needed)
CMD ["/bin/bash"]