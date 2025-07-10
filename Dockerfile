FROM python:3.10-slim as python-base

FROM ollama/ollama:latest

COPY --from=python-base /usr/local /usr/local

RUN apt-get update && \
    apt-get install -y libpython3.10 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip3

RUN pip3 install langchain langchain-ollama

RUN ollama serve & \
    sleep 10 && \
    ollama pull gemma3n && \
    pkill ollama

WORKDIR /workspace

RUN python3 -c "import sys; print(f'Python version: {sys.version}'); import langchain; from langchain_ollama import OllamaLLM; print('Setup verified successfully!')"