FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY ad_review_env/pyproject.toml .
COPY ad_review_env/__init__.py .
COPY ad_review_env/models.py .
COPY ad_review_env/data.py .
COPY ad_review_env/grader.py .
COPY ad_review_env/agent.py .
COPY ad_review_env/client.py .
COPY ad_review_env/baseline.py .
COPY ad_review_env/server/ ./server/
COPY openenv.yaml .
COPY inference.py .

# Force cache bust on code changes
RUN echo "build-v7"

RUN uv pip install --system -e ".[inference]" || pip install openenv-core openai requests

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
