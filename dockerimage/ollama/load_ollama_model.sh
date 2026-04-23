#!/usr/bin/env bash
set -euo pipefail

# If file was mounted with CRLF line endings, remove them in-place (requires sed available in image)
if grep -q $'\r' "/load_ollama_model.sh" 2>/dev/null || true; then
  sed -i 's/\r$//' "/load_ollama_model.sh" || true
fi

# Start ollama in background, pull the configured model, then signal readiness
/bin/ollama serve &
pid=$!
sleep 5
echo "Starting pulling model $OLLAMA_MODEL..."
ollama pull "$OLLAMA_MODEL"
echo "DONE! Pulling model $OLLAMA_MODEL successful!"

if [ -n "${OLLAMA_EMBED_MODELS:-}" ]; then
  IFS=',' read -ra embed_models <<< "$OLLAMA_EMBED_MODELS"
  for model in "${embed_models[@]}"; do
    model_trimmed="$(echo "$model" | sed 's/^ *//;s/ *$//')"
    if [ -n "$model_trimmed" ]; then
      echo "Starting pulling embedding model $model_trimmed..."
      ollama pull "$model_trimmed"
      echo "DONE! Pulling embedding model $model_trimmed successful!"
    fi
  done
elif [ -n "${OLLAMA_EMBED:-}" ]; then
  echo "Starting pulling embedding model $OLLAMA_EMBED..."
  ollama pull "$OLLAMA_EMBED"
  echo "DONE! Pulling embedding model $OLLAMA_EMBED successful!"
fi

touch /tmp/ollama_is_ready
wait "$pid"

