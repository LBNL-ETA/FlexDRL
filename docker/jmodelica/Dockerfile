# This dockerfile is adapted from https://github.com/lbl-srg/docker-ubuntu-jmodelica.
# The difference is that an ENV variable is added to indicate the JModelica version.

FROM der_base

USER root

# Set environment variables
ENV SRC_DIR /usr/local/src
ENV MODELICAPATH /usr/local/JModelica/ThirdParty/MSL
ENV VER 2_2

# Avoid warnings
# debconf: unable to initialize frontend: Dialog
# debconf: (TERM is not set, so the dialog frontend is not usable.)
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install required packages
RUN apt-get update && \
    apt-get install -y \
    ant \
    autoconf \
    cmake \
    cython \
    default-jdk \
    g++ \
    gfortran \
    ipython \
    jcc \
    libboost-dev \
    openjdk-8-jdk \
    pkg-config \
    python-dev \
    python-jpype \
    python-lxml \
    python-nose \
    python-scipy \
    python-tk \
    subversion \
    swig \
    wget \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Install jcc-3.0 to avoid error in python -c "import jcc"
RUN python -m pip install --upgrade pip
RUN ln -s /usr/lib/jvm/java-8-openjdk-amd64 /usr/lib/jvm/java-8-oracle && \
    python -m pip install --upgrade jcc

RUN export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
RUN export JCC_JDK=/usr/lib/jvm/java-8-openjdk-amd64

# Copy dependency HSL MA27
# COPY dependencies/coinhsl-archive-2014.01.17 $SRC_DIR/Ipopt-3.12.4/ThirdParty/HSL/coinhsl
# Get Install Ipopt and JModelica, and delete source code with is more than 1GB large
RUN cd $SRC_DIR && \
    wget wget -O - http://www.coin-or.org/download/source/Ipopt/Ipopt-3.12.4.tgz | tar xzf - && \
    cd $SRC_DIR/Ipopt-3.12.4/ThirdParty/Blas && \
    ./get.Blas && \
    cd $SRC_DIR/Ipopt-3.12.4/ThirdParty/Lapack && \
    ./get.Lapack && \
    cd $SRC_DIR/Ipopt-3.12.4/ThirdParty/Mumps && \
    ./get.Mumps && \
    cd $SRC_DIR/Ipopt-3.12.4/ThirdParty/Metis && \
    ./get.Metis && \
    cd $SRC_DIR/Ipopt-3.12.4 && \
    ./configure --prefix=/usr/local/Ipopt-3.12.4 && \
    make install && \
    cd $SRC_DIR && \
    svn export https://svn.jmodelica.org/tags/2.2 JModelica && \
    cd $SRC_DIR/JModelica/external && \
    rm -rf $SRC_DIR/JModelica/external/Assimulo && \
    svn export https://svn.jmodelica.org/assimulo/trunk Assimulo && \
    cd $SRC_DIR/JModelica && \
    rm -rf build && \
    mkdir build && \
    cd $SRC_DIR/JModelica/build && \
    ../configure --with-ipopt=/usr/local/Ipopt-3.12.4 --prefix=/usr/local/JModelica && \
    make install && \
    make casadi_interface && \
    rm -rf $SRC_DIR

USER developer

# Avoid warning that Matplotlib is building the font cache using fc-list. This may take a moment.
# This needs to be towards the end of the script as the command writes data to
# /home/developer/.cache
RUN python -c "import matplotlib.pyplot"

WORKDIR ${HOME}

# Set the environment variable
ENV ROOT_DIR /usr/local
ENV JMODELICA_HOME ${ROOT_DIR}/JModelica
ENV PYTHONPATH ${PYTHONPATH}:${JMODELICA_HOME}/Python

