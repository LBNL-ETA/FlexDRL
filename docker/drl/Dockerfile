# This dockerfile creates user, and installs python

FROM der_eplus:8.5

# Install required python packages with root

USER root
# ENV HOME /home/developer


# Install base python packages
RUN python -m pip install --upgrade pip && \
    python -m pip install --user \
	torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html  \
	tensorboardX \
	tqdm \
	SQLAlchemy \
	psycopg2-binary

# ENV PATH="/home/developer/.local/lib/python2.7/site-packages:${PATH}"
# ENV PYTHONPATH="/home/developer/.local/lib/python2.7/site-packages:${PYTHONPATH}"

# WORKDIR ${HOME}
# =======
# RUN python -m pip install --user torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# RUN python -m pip install --user tensorboardX

# RUN python -m pip install --user tqdm

USER root

WORKDIR ${HOME}





