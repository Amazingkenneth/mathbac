# Sandbox image for reproducing Cha et al. 2022.
# Pins follow the Code Ocean capsule's environment (capsule 3253425) so results are comparable.
#
#   docker build -t mathbac-cha2022:latest -f env/Dockerfile .
#   apptainer build env/mathbac-cha2022.sif docker-daemon://mathbac-cha2022:latest   # for Slurm
#
# GHECOM is distributed by PDBj (https://pdbj.org/ghecom). Download the source tarball and pass
# its path with --build-arg GHECOM_TARBALL=env/vendor/ghecom-src.tar.gz (the file is not committed).

FROM continuumio/miniconda3:4.9.2

ARG DEBIAN_FRONTEND=noninteractive
ARG GHECOM_TARBALL=env/vendor/ghecom-src.tar.gz
ARG UNIFIED_FAST_COMMIT=04b567b4ef85dc916209510bd830b1d081e5beb5

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y -c conda-forge python=3.8 pymol-open-source=2.5.0 \
        jupyter=1.0.0 matplotlib=3.3.4 pandas=1.3.3 scipy=1.4.1 seaborn=0.11.2 \
    && conda clean -ya

RUN pip install --no-cache-dir \
        biopandas==0.2.9 biopython==1.79 cython==0.29.24 networkx==2.6.3 numpy==1.21.2 \
        powerlaw==1.5 prody==2.0 rampy==0.4.9 scikit-learn==1.0 tensorflow==2.3.0 xgboost==1.4.2 \
        python-igraph==0.9.8 pyyaml==6.0 \
    && pip install --no-cache-dir graphriccicurvature==0.5.3

# GHECOM (geometric descriptors: shell volume, R_inacc, pocketness)
COPY ${GHECOM_TARBALL} /tmp/ghecom.tar.gz
RUN mkdir -p /opt/ghecom && tar -xzf /tmp/ghecom.tar.gz -C /opt/ghecom --strip-components=1 \
    && cd /opt/ghecom/src && make && cp ghecom /usr/local/bin/ && rm /tmp/ghecom.tar.gz

# Released code: vectorized descriptors from Jessica Ma, pinned
RUN git clone https://github.com/jessqma/unified_fast /opt/unified_fast \
    && cd /opt/unified_fast && git checkout ${UNIFIED_FAST_COMMIT}

# Thin wrappers that the tool cards invoke, and their smoke tests
COPY env/tools /opt/mathbac/tools
RUN chmod +x /opt/mathbac/tools/*.sh

WORKDIR /work
CMD ["sleep", "infinity"]
