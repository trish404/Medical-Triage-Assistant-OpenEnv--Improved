FROM python:3.11-slim

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY medical_triage_env.py .
COPY main.py .
COPY inference.py .
COPY dashboard.html .
COPY openenv.yaml .
COPY README.md .

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import requests; r = requests.get('http://localhost:7860/health', timeout=5); exit(0 if r.status_code == 200 else 1)"

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]