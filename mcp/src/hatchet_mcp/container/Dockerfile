# syntax=docker/dockerfile:1.7
# hatchet-mcp runtime image: HATCHet built from source inside its own conda environment.
#
# Build arguments
#   HATCHET_SOURCE       git (fetch HATCHET_REPO at HATCHET_COMMIT) or local (copy the
#                        named build context "hatchet_src")
#   HATCHET_REPO         git URL to fetch from
#   HATCHET_REF          branch or tag, recorded as a label
#   HATCHET_COMMIT       commit to check out, recorded as a label
#   HATCHET_MCP_VERSION  hatchet-mcp version, recorded as a label
#
# The conda environment is created from environment.yaml alone, so it is rebuilt only
# when that file changes, not on every source change. A Gurobi licence is never baked
# into the image; hatchet-mcp mounts it read-only at run time.

ARG HATCHET_SOURCE=git

FROM condaforge/miniforge3:26.7.2-0 AS base
RUN apt-get update \
 && apt-get install -y --no-install-recommends git poppler-utils \
 && rm -rf /var/lib/apt/lists/*

FROM base AS source-git
ARG HATCHET_REPO
ARG HATCHET_COMMIT
RUN git init -q /opt/hatchet \
 && git -C /opt/hatchet remote add origin "${HATCHET_REPO}" \
 && git -C /opt/hatchet fetch -q --depth 1 origin "${HATCHET_COMMIT}" \
 && git -C /opt/hatchet checkout -q FETCH_HEAD \
 && rm -rf /opt/hatchet/.git

FROM base AS source-local
COPY --from=hatchet_src . /opt/hatchet

FROM source-${HATCHET_SOURCE} AS source

FROM base AS env
COPY --from=source /opt/hatchet/environment.yaml /tmp/environment.yaml
RUN mamba env create -y -p /opt/env -f /tmp/environment.yaml \
 && mamba clean -afy

FROM env AS final
ENV PATH=/opt/env/bin:${PATH} \
    CONDA_PREFIX=/opt/env

COPY --from=source /opt/hatchet /opt/hatchet
RUN cd /opt/hatchet \
 && pip install --no-cache-dir --no-build-isolation --no-deps . \
 && hatchet --version \
 && python -c "import hatchet.cluster_bins.hmm as h; assert h._USE_CPP, 'C++ HMM backend not built'"

# NB: containers run as the host uid, so the source tree (incl. data/reference) must be world-readable
RUN chmod -R a+rX /opt/hatchet

ARG HATCHET_REPO
ARG HATCHET_REF
ARG HATCHET_COMMIT
ARG HATCHET_MCP_VERSION
LABEL org.opencontainers.image.title="hatchet-mcp" \
      org.opencontainers.image.source="${HATCHET_REPO}" \
      hatchet.ref="${HATCHET_REF}" \
      hatchet.commit="${HATCHET_COMMIT}" \
      hatchet_mcp.version="${HATCHET_MCP_VERSION}"

WORKDIR /job
