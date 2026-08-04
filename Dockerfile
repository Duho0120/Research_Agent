FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RESEARCH_AGENT_RUNTIME_DIR=/mnt/research-agent/runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt

COPY research_agent /app/research_agent
COPY scripts /app/scripts
COPY configs /app/configs
COPY docs /app/docs
COPY README.md README_DEMO.md /app/
COPY scripts/docker_entrypoint.sh /app/scripts/docker_entrypoint.sh
COPY competitions/titanic /app/demo_seed/competitions/titanic
COPY experiments/titanic /app/demo_seed/experiments/titanic
COPY runs/titanic /app/demo_seed/runs/titanic
COPY memory/titanic /app/demo_seed/memory/titanic
COPY demo_workspaces/titanic /app/demo_seed/demo_workspaces/titanic
COPY submissions/titanic /app/demo_seed/submissions/titanic

RUN mkdir -p /mnt/research-agent/runtime \
    /mnt/research-agent/demo_workspaces \
    /mnt/research-agent/experiments \
    /mnt/research-agent/memory \
    /mnt/research-agent/submissions
RUN chmod +x /app/scripts/docker_entrypoint.sh

ENTRYPOINT ["/app/scripts/docker_entrypoint.sh"]
EXPOSE 8080
CMD ["python", "-B", "-m", "research_agent.web_app"]
