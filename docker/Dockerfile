FROM ubuntu:22.04

# Set default shell during Docker image build to bash
SHELL ["/bin/bash", "-l", "-c"]

# Copy docker clean script
COPY docker/install/docker_clean.sh /docker_clean.sh
RUN chmod +x /docker_clean.sh

# Install base packages
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
	apt-get -y upgrade && \
	apt-get install --no-install-recommends -y \
		sudo \
		locales \
		git \
    texlive-latex-extra \
    texlive-bibtex-extra \
    latexmk \
    python3-pip \
    python3-venv \
    graphviz \
		&& \
	/docker_clean.sh

# Initialise system locale
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
RUN locale-gen en_US.UTF-8

# create a user for running the container
ARG UID_USER=1000
RUN sudo useradd --create-home -l -u $UID_USER -G sudo,plugdev user && \
  echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER user

COPY requirements.txt /tmp
RUN python3 -m venv ~/.venv && \
  source ~/.venv/bin/activate && \
  python -m pip install -r /tmp/requirements.txt

# setup entry point
COPY docker/install/user_setup.sh /tmp
RUN sudo chmod +x /tmp/user_setup.sh && \
  /tmp/user_setup.sh

# create workdir, this is where the source code will be mounted
VOLUME /workdir
WORKDIR /workdir
RUN sudo mkdir -p /workdir && sudo chown -R user:user /workdir

# setup entry point
COPY docker/install/entrypoint.sh /
RUN sudo chmod +x /entrypoint.sh

CMD ["/bin/bash" , "-l"]
