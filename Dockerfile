FROM python:3.11-slim

# system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install python deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY src /app/src

# entrypoint script (gradio app)
# assume you will have src/main.py that launches the gradio UI
# e.g. in src/main.py:
#   from src.rl_budget.gradio_app import build_interface
#   app = build_interface()
#   app.launch(server_name="0.0.0.0", server_port=7860)
COPY src/main.py /app/src/main.py

EXPOSE 7860   # gradio
EXPOSE 8000   # prometheus_client if you start it

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.main"]
