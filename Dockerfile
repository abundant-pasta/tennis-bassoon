FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml requirements.lock README.md /app/
COPY src /app/src
COPY scripts /app/scripts
COPY config_tennis.py run_pipeline.py /app/
COPY data/tennis/features/matches.parquet /opt/tennis-seed/data/tennis/features/matches.parquet
COPY data/tennis/features/rankings.parquet /opt/tennis-seed/data/tennis/features/rankings.parquet
COPY data/tennis/model /opt/tennis-seed/data/tennis/model
COPY data/tennis/raw/atp_players.csv /opt/tennis-seed/data/tennis/raw/atp_players.csv

RUN python -m pip install --upgrade pip && \
    python -m pip install . && \
    chmod +x /app/scripts/*.sh /app/scripts/*.py

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["tennis-daily-run"]
