FROM python:3.12-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY output/ ./output/

# Set environment
ENV PYTHONUNBUFFERED=1

# Run the pipeline
CMD ["python3", "src/orchestration/prefect_flow.py"]