FROM continuumio/miniconda3

WORKDIR /app

#Install dependencies
RUN apt-get update && apt-get install -yq git cmake build-essential \
  libgl1-mesa-dev libsdl2-dev \
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
  libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
  libsdl-sge-dev python3-pip

#Add current folder to docker container
ADD . .

# Create the environment:
RUN conda env create -f environment.yml
ENV PATH /opt/conda/envs/mro_env/bin:$PATH
RUN /bin/bash -c "source activate XRL"

#Install google football environment
RUN cd /gfootball_env && pip3 install .
WORKDIR '/'


# The code to run when container is started:
CMD ["test.sh" ]