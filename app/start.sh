set -e

uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "[start] waiting for the API to load the MLflow model ..."
for i in $(seq 1 120); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" 2>/dev/null; then
    echo "[start] API is ready"
    break
  fi
  sleep 2
done

# keep the API visible in the container logs
tail --pid=$API_PID -f /dev/null &

exec streamlit run dashboard.py --server.address 0.0.0.0 --server.port "${PORT:-8080}" --server.headless true --browser.gatherUsageStats false