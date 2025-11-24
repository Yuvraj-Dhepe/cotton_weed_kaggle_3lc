#!/bin/bash

# Script to start Optuna Dashboard and MLflow UI in the background

# Ensure optuna-dashboard is installed
if ! command -v optuna-dashboard &> /dev/null && ! uv run command -v optuna-dashboard &> /dev/null; then
    echo "Installing optuna-dashboard..."
    uv add optuna-dashboard
fi

echo "==================================================="
echo "STARTING DASHBOARDS"
echo "==================================================="

OPTUNA_PORT=8080
MLFLOW_PORT=5000

# Ensure Optuna database exists
mkdir -p temp
if [ ! -f "temp/optuna.db" ]; then
    echo "Initializing Optuna database..."
    uv run optuna create-study --storage sqlite:///temp/optuna.db --study-name cotton_weed_optimization --direction maximize
fi

# 1. Start Optuna Dashboard
# Connects to the sqlite database created by the optimization script
echo "[1/2] Starting Optuna Dashboard..."
nohup uv run optuna-dashboard sqlite:///temp/optuna.db --port $OPTUNA_PORT > temp/optuna_dashboard.log 2>&1 &
OPTUNA_PID=$!
echo "Optuna Dashboard running at http://127.0.0.1:$OPTUNA_PORT (PID: $OPTUNA_PID)"

# 2. Start MLflow UI
# Points to the 'runs/mlflow' directory where YOLO logs experiments
echo "[2/2] Starting MLflow UI..."
nohup uv run mlflow ui --backend-store-uri runs/mlflow --port $MLFLOW_PORT > temp/mlflow_ui.log 2>&1 &
MLFLOW_PID=$!
echo "MLflow UI running at http://127.0.0.1:$MLFLOW_PORT (PID: $MLFLOW_PID)"

echo "==================================================="
echo "Logs are being written to:"
echo "  - optuna_dashboard.log"
echo "  - mlflow_ui.log"
echo ""
echo "To stop these processes, run:"
echo "kill $OPTUNA_PID $MLFLOW_PID"
echo "==================================================="
