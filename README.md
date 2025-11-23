### Setup 
uv pip install git+https://github.com/3lc-ai/3lc-examples

### Environment Setup
uv init
uv pip install git+https://github.com/3lc-ai/3lc-ultralytics@develop --no-sources

```
nohup mlflow server --backend-store-uri runs/mlflow > mlflow.log 2>&1 &
uv run yolo settings mlflow=True
```