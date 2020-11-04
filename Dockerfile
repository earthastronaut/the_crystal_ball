# Build from official jupyter image
# https://hub.docker.com/r/jupyter/scipy-notebook/tags
# tag: 95ccda3619d0 2020-10-31
FROM jupyter/scipy-notebook:95ccda3619d0 as base

# Directory for source code
WORKDIR /usr/src/app

# System Dependencies
# Install gcc required for python dep xgboost
RUN conda update -n base conda \
  && conda install -y libgcc-ng

# Add python requirements and install
COPY requirements.txt .
RUN ${CONDA_DIR}/bin/pip install -U pip \
  && ${CONDA_DIR}/bin/pip install --no-cache-dir -r requirements.txt \
  ``

# Add source code for the project
COPY the_crystal_ball scripts ./

# Add to scripts and source to path
ENV PYTHONPATH=/usr/src/app
ENV PATH=/usr/src/app/scripts:$PATH

# ############################################################################ #
# Dev Build
# ############################################################################ #

FROM base as dev 

# Add development requirements and install
COPY requirements_dev.txt ./
RUN pip install --no-cache-dir -r requirements_dev.txt

# Create interactive work path
WORKDIR /work

# Add to scripts and source to path
ENV PYTHONPATH=/work
ENV PATH=/work/scripts:$PATH

CMD ["jupyter", "lab", "--port=8888"]

# ############################################################################ #
# Release Build
# ############################################################################ #

FROM base as release

CMD ["/bin/bash"]
