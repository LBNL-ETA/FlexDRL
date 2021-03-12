# This dockerfile creates user, and installs python

FROM ubuntu:16.04

# Install required packages with root
RUN apt-get update && apt-get install -y \
    python-pip \
    python-dev \
    git

# Create non-root user
# Replace 1000 with your user / group id
RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    mkdir -p /etc/sudoers.d && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer
USER developer
ENV HOME /home/developer

# Install base python packages
RUN python -m pip install --upgrade pip && \
    python -m pip install --user \
    numpy==1.15.2 \
    pandas==0.23.4 \
    matplotlib \
    python-dateutil==2.6.1 \
	pytz==2017.2 \
	scikit-learn==0.18.2 \
	sphinx==1.6.3 \
	numpydoc==0.7.0 \
	tzwhere==2.3 \
	influxdb==5.2.2 \
	pyyaml==3.13 \
	gym \
	requests==2.21.0 \
	pydoe

# Add repo to pythonpath (will be in /mnt/shared/)
ENV PYTHONPATH ${PYTHONPATH}:/mnt/shared

WORKDIR ${HOME}


