FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY __init__.py .
COPY models.py .
COPY data.py .
COPY grader.py .
COPY agent.py .
COPY client.py .
COPY server/ ./server/
COPY README.md .

# Install dependencies
RUN uv pip install --system -e ".[core]" || pip install openenv-core

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
