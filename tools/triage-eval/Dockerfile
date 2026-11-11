# Agent container: the Alertmanager -> agent webhook receiver. Runs as a K8s Deployment
# (k8s/deployment.yaml), never a sidecar.
#
# Build from the repo root (paths below are relative to it, not this directory), so the image
# can also pull in adapters/harness modules that will later live alongside this one:
#   docker build -f tools/triage-eval/Dockerfile -t sslm-agent:local .
FROM python:3.12-slim

WORKDIR /app

# The agent's tools are argv-exec'd binaries (harness/envelope.py): kubectl and wget at the top
# level. python:3.12-slim ships neither, so until 2026-10-15 every command the agent ran in the
# earlier model rounds and the API arm round died with FileNotFoundError (exit=-1) and the model triaged from the
# alert text alone. kubectl pinned to the kind node version (kindest/node:v1.36.1), hash checked.
ARG KUBECTL_VERSION=v1.36.1
ARG TARGETARCH=amd64
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && case "$TARGETARCH" in \
         amd64) sum=629d3f410e09bf49b64ae7079f7f0bda1191efed311f7d37fdbab0ad5b0ec2b7 ;; \
         arm64) sum=59f7ee8e477fae658447607dc3c8790ac17a1b016c01c622c12070e969e2d4e7 ;; \
         *) echo "unsupported arch $TARGETARCH" >&2; exit 1 ;; \
       esac \
    && wget -q -O /usr/local/bin/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl" \
    && echo "$sum  /usr/local/bin/kubectl" | sha256sum -c - \
    && chmod 0755 /usr/local/bin/kubectl

# requirements-agent.txt only -- never requirements.txt (mlx-lm has no Linux wheel; see that
# file's own comment). Copied and installed before the source so a source-only change reuses
# this layer.
COPY tools/triage-eval/requirements-agent.txt ./requirements-agent.txt
RUN pip install --no-cache-dir -r requirements-agent.txt

COPY tools/triage-eval/harness ./harness
COPY tools/triage-eval/adapters ./adapters
COPY tools/triage-eval/configs ./configs
COPY tools/triage-eval/entrypoint.py ./entrypoint.py
# Comparator arm (COMPARATOR_ARM=1): entrypoint.py builds the Anthropic client through
# cloud_adapter.make_anthropic_model_client, so the pod and run_matrix.py share one set of checks.
COPY tools/triage-eval/cloud_adapter.py ./cloud_adapter.py

# Non-root: the Deployment's RBAC (k8s/) already scopes what the ServiceAccount
# may do against the apiserver; running as root inside the container would still be a wider
# blast radius than the envelope needs for a process that only serves HTTP and shells out to
# `kubectl`/`wget` (allowlisted binaries, harness/envelope.py) it does not itself own.
RUN useradd --no-create-home --uid 10001 sslm-agent
USER sslm-agent

EXPOSE 8080

ENTRYPOINT ["python3", "entrypoint.py"]
