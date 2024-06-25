FROM python:3.10

# debian setting
ENV DEBIAN_FRONTEND=noninteractive \
    DEBIAN_FRONTEND=teletype \
    DEBCONF_NOWARNINGS=yes
# python pyc setting
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /root
RUN pip3 install -U pip==23.2.1
RUN pip3 install \
    openai==1.23.6 \
    azure-storage-blob==12.19.1 \
    azure-cognitiveservices-vision-computervision==0.9.0 \
    langchain==0.1.16 \
    langchain-openai==0.1.3 \
    langchainhub==0.1.15 \
    langchain_experimental==0.0.57 \
    langgraph==0.0.48 \
    retry==0.9.2 \
    portalocker==2.8.2
COPY / /root/

# Set up the Bash shell environment
RUN echo "export PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /etc/bash.bashrc
RUN echo "export PS1='\[\033[01;31m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '" >> /root/.bashrc

# Create a user; it must not be ROOT and its UID should be greater than 1000.
RUN groupadd --gid 2100 "user" && \
    useradd --no-log-init --shell "/bin/bash" --uid 1100 --gid 2100 "user"
USER user
WORKDIR /home

ENTRYPOINT []

CMD ["/bin/bash"]
