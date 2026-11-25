# Deployment Guide
*The bash commands below are for Windows PowerShell. If you are using a different shell, please adjust the commands accordingly.*

## Local Deployment
Build the Docker image and run the container locally using the following commands:

```bash
docker build -t wine-quality-predictor .
```

To run the container, you can choose between two options:

1. With controlled dummy data:
```bash
docker run --rm -d --name wine-quality-predictor `
  -p 8000:8000 `
  -e MODEL_SOURCE=dummy `
  -e DATABASE_PATH=/app/var/app.db `
  wine-quality-predictor:local
```

2. With real data (make sure the env variables are set in the `.env` file):
```bash
docker run --rm -d --name wine-quality-predictor `
  -p 8000:8000 `
  --env-file .env `
  wine-quality-predictor:local
```



