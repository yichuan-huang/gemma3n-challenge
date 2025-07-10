FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y \
    curl \
    ca-certificates \
    procps \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN pip install ollama langchain langchain-ollama pillow

RUN ollama serve & \
    sleep 10 && \
    ollama pull gemma3n:e2b && \ 
    pkill ollama

WORKDIR /workspace

RUN python3 -c "import sys; print(f'Python version: {sys.version}'); import langchain; from langchain_ollama import OllamaLLM; print('Setup verified successfully!')"
CMD ["ollama" , "serve"]