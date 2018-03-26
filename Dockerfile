# To Build:
# docker build -t training -f Dockerfile .

# To run:
# docker run -it training

FROM ubuntu

RUN apt-get -y update && apt-get -y install vim
RUN apt-get -y update --fix-missing && \
    apt-get install -y \
        python-pip \
        python-dev \
        libev4 \
        libev-dev \
        gcc \
        libxslt-dev \
        libxml2-dev \
        libffi-dev \
        python-numpy \
        python-scipy && \
    pip install --upgrade pip && \
    pip install scikit-learn flask-restful && \
    pip install python-crfsuite gensim nltk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#WORKDIR /workspace
#RUN chmod -R a+w /workspace

# Add code to folder
RUN mkdir -p /src
ADD code/ /src/ontology-ner
ENV PYTHONPATH='$PYTHONPATH:/src/ontology-ner'

#EXPOSE 8888
#ENTRYPOINT ["/bin/bash", "-c", "jupyter notebook --ip='*' --allow-root --no-browser --port=8888"]
ENTRYPOINT ["/bin/bash"]