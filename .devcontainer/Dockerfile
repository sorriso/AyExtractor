# =============================================================================
# Dockerfile for Ubuntu Dev Environment
# Version: 1.2.0
# Description: Development container with Python 3.12, Node.js LTS, Docker tools
# =============================================================================

# Use Ubuntu Server minimal image (lightest official Ubuntu)
ARG TARGETOS=linux
ARG TARGETARCH=amd64
FROM --platform=${TARGETOS}/${TARGETARCH} ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

USER root

# Single apt layer; add sudo; keep recommends off; clean properly
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git nano unzip ca-certificates gnupg lsb-release net-tools iputils-ping \
    software-properties-common build-essential \
    sudo qemu-system-x86 qemu-utils jq \
    # keep the stock python tools for bootstrapping only
    python3-pip python3-venv \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev telnet \
 && apt-get install -y wget xorriso squashfs-tools file \
 && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install kubectl (latest stable)
RUN curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
 && chmod +x kubectl && mv kubectl /usr/local/bin/

# Install nerdctl (latest release)
RUN set -eux; \
    NERDCTL_TAG="$(curl -s https://api.github.com/repos/containerd/nerdctl/releases/latest \
      | awk -F '\"' '/tag_name/ {print $4; exit}')" ; \
    NERDCTL_VER="${NERDCTL_TAG#v}"; \
    curl -L "https://github.com/containerd/nerdctl/releases/download/${NERDCTL_TAG}/nerdctl-${NERDCTL_VER}-linux-amd64.tar.gz" -o /tmp/nerdctl.tgz; \
    tar -xzf /tmp/nerdctl.tgz -C /usr/local/bin/ nerdctl; \
    rm /tmp/nerdctl.tgz; \
    nerdctl --version || true

# Install HashiCorp Packer (repo + package)
RUN curl -fsSL https://apt.releases.hashicorp.com/gpg | gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" > /etc/apt/sources.list.d/hashicorp.list \
 && apt-get update && apt-get install -y --no-install-recommends packer \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Node.js (latest LTS)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Make Python 3.12 the default for python3 and python (priority required)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 2

# Ensure pip is for Python 3.12 (use get-pip for reliability)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
 && python3.12 /tmp/get-pip.py \
 && rm -f /tmp/get-pip.py

# Global tools
RUN npm install -g @anthropic-ai/claude-code \
 && python3.12 -m pip install --no-cache-dir python-dotenv requests pyyaml

# Create docker group with common GID (999) if not exists
RUN getent group docker || groupadd -g 999 docker

# Create a non-root user for development
RUN useradd -m -s /bin/bash developer \
 && usermod -aG sudo developer \
 && usermod -aG docker developer \
 && echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN python3.12 -m pip install --upgrade pip setuptools wheel

# Copy Python dependencies from backend directory
COPY pyproject.toml .

# Install Python dependencies
# RUN pip install --no-cache-dir -e ".[all]"


USER developer

WORKDIR /workspace

CMD ["/bin/bash"]