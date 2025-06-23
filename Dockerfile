# syntax=docker/dockerfile:1.4
# Enable BuildKit for better caching

# Multi-stage build for smaller image size
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Cache mount for pip packages (using fast CPU-only requirements)
COPY requirements_fast.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user --no-cache-dir -r /tmp/requirements_fast.txt

# Final stage - minimal runtime image
FROM python:3.11-slim

# Copy only the installed packages from builder
COPY --from=builder /root/.local /root/.local

# Add user packages to PATH
ENV PATH=/root/.local/bin:$PATH

# Set working directory
WORKDIR /usr/src/app

# Create necessary directories first
RUN mkdir -p cache explain/actions explain/mega_explainer data configs templates static

# Layer 1: Copy static assets and configs (rarely change)
COPY --link templates/ templates/
COPY --link static/ static/
COPY --link configs/diabetes-gpt4-config.gin configs/

# Layer 2: Copy data files (rarely change)
COPY --link data/*.pkl data/
COPY --link data/*.csv data/

# Layer 3: Copy stable core modules (change occasionally)
COPY --link explain/__init__.py explain/
COPY --link explain/utils.py explain/
COPY --link explain/conversation.py explain/
COPY --link explain/dataset_description.py explain/
COPY --link explain/feature_interaction.py explain/
COPY --link explain/explanation.py explain/
COPY --link explain/write_to_log.py explain/

# Layer 4: Copy action modules (change occasionally)  
COPY --link explain/actions/ explain/actions/
COPY --link explain/mega_explainer/ explain/mega_explainer/
COPY --link explain/action.py explain/

# Layer 5: Copy frequently changing files last
COPY --link explain/enhanced_logic.py explain/
COPY --link explain/gpt4_decoder.py explain/
COPY --link explain/autogen_decoder.py explain/
COPY --link explain/smart_action_dispatcher.py explain/
COPY --link explain/feature_mappings.py explain/
COPY --link explain/lazy_mega_explainer.py explain/
COPY --link explain/generalized_agents.py explain/
COPY --link flask_app_gpt4.py ./

# Set environment variables
ENV OPENAI_API_KEY=""
ENV FLASK_APP=flask_app_gpt4.py
ENV PYTHONPATH=/usr/src/app
ENV USE_AUTOGEN=true

# Expose port
EXPOSE 4455

# Run command
CMD ["python", "flask_app_gpt4.py"]
