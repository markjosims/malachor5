# Made with the help of Her Computational Overlordliness, ChatGPT
FROM ubuntu:24.04
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set up a virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install dependencies in the virtual environment
COPY requirements/requirements.txt .
RUN pip install --upgrade pip
RUN pip install setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Set entrypoint
CMD ["/bin/bash"]
