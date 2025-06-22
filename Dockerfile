# Multi-stage build for smaller image size
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_gpt4.txt /tmp/
RUN pip install --user --no-cache-dir -r /tmp/requirements_gpt4.txt

# Final stage - minimal runtime image
FROM python:3.11-slim

# Copy only the installed packages from builder
COPY --from=builder /root/.local /root/.local

# Add user packages to PATH
ENV PATH=/root/.local/bin:$PATH

# Set working directory
WORKDIR /usr/src/app

# Copy only essential files for GPT-4 version
COPY flask_app_gpt4.py ./
COPY explain/__init__.py explain/
COPY explain/enhanced_logic.py explain/
COPY explain/simple_intent_decoder.py explain/
COPY explain/gpt4_decoder.py explain/
COPY explain/action.py explain/
COPY explain/conversation.py explain/
COPY explain/dataset_description.py explain/
COPY explain/feature_interaction.py explain/
COPY explain/explanation.py explain/
COPY explain/write_to_log.py explain/
COPY explain/actions/ explain/actions/
COPY explain/mega_explainer/ explain/mega_explainer/
COPY explain/prompts/ explain/prompts/
COPY explain/utils.py explain/
COPY explain/prompts.py explain/
COPY explain/sample_prompts_by_action.py explain/
COPY data/*.pkl data/
COPY data/*.csv data/
COPY configs/diabetes-gpt4-config.gin configs/
COPY templates/ templates/
COPY static/ static/

# Create necessary directories
RUN mkdir -p cache logs

# Set environment variables
ENV OPENAI_API_KEY=""
ENV FLASK_APP=flask_app_gpt4.py
ENV PYTHONPATH=/usr/src/app

# Expose port
EXPOSE 4455

# Run command
CMD ["python", "flask_app_gpt4.py"]
