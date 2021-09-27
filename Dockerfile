FROM  r-base:4.0.1

ARG HOME

# Install system dependencies
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    apt-transport-https \
    build-essential \
    wget \
    curl \
    git \
    cmake \
    gfortran \
    libatlas-base-dev \
    libbz2-dev \
    libcurl4-openssl-dev \
    libicu-dev \
    liblzma-dev \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libpcre3-dev \
    libtcl8.6 \
    libtiff5 \
    libtk8.6 \
    libx11-6 \
    libxt6 \
    locales \
    tzdata \
    zlib1g-dev \
    libffi-dev \
    libsqlite3-dev \
    libreadline-dev \
    texlive-xetex \ 
    texlive-fonts-recommended \
    texlive-generic-recommended \
    python3-pip

# Install system dependencies for the tidyverse R packages
RUN apt-get install -y \
    make \
    libcurl4-openssl-dev \
    libssl-dev \
    pandoc \
    libxml2-dev \
    r-cran-rgl

WORKDIR ${HOME}

# copy file with packages requeriments
COPY requeriments.R ${HOME}requeriments.R
COPY requeriments.txt ${HOME}requeriments.txt

# install pyenv and python
#RUN git clone git://github.com/yyuu/pyenv.git .pyenv

#ENV PYENV_ROOT ${HOME}/.pyenv
#ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

#RUN pyenv install 3.8.1
#RUN pyenv global 3.8.1
#RUN pyenv rehash

RUN pip  install --upgrade pip 

## install python packages
RUN pip install -r requeriments.txt

## install R-packages
RUN Rscript requeriments.R

# remove files
RUN rm requeriments.R
RUN rm requeriments.txt