FROM python:3.10-slim

# ------------------------------------------------------------
# set working directory
# ------------------------------------------------------------
WORKDIR /app

# ------------------------------------------------------------
# install system dependencies 
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# install Python dependencies
# ------------------------------------------------------------
COPY requirements.txt .
RUN pip install -r requirements.txt

# ------------------------------------------------------------
# Default command (can be overridden for arguments passing)
# ------------------------------------------------------------
CMD ["python", "run_benchmark.py"]