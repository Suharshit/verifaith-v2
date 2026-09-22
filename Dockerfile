FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models
WORKDIR /app

# CPU-only torch keeps the image small; use a CUDA base image for GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[service,nli]"

# Bake the NLI model into the image so containers start fast and offline.
ARG NLI_MODEL=cross-encoder/nli-deberta-v3-base
ENV VERIFAITH_NLI_MODEL=${NLI_MODEL}
RUN python -c "from verifaith.nli.hf import HFEntailmentModel as M; M('${NLI_MODEL}')"

RUN useradd -m app && chown -R app /models
USER app
EXPOSE 8000
CMD ["uvicorn", "verifaith.service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
