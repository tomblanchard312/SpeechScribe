# Docker Deployment Complete Setup Guide

## Overview

This guide provides a complete Docker deployment setup for SpeechScribe with:
- ✅ GPU acceleration (NVIDIA CUDA)
- ✅ Ollama LLM integration
- ✅ Persistent volumes for models and plugins
- ✅ Production-ready configuration
- ✅ Development mode with hot-reload
- ✅ Comprehensive testing and monitoring

## Quick Start (Less than 5 minutes)

### 1. Windows Users
```bash
cd docker
quick-start.bat
```

### 2. Linux/macOS Users
```bash
cd docker
chmod +x quick-start.sh
./quick-start.sh
```

The scripts will automatically:
- Check prerequisites
- Detect GPU support
- Create `.env` configuration
- Start all services
- Verify service health
- Show access URLs

## Manual Setup

### Prerequisites
- Docker 20.10+
- Docker Compose 1.29+
- NVIDIA Docker Runtime (for GPU support)
- 8GB RAM minimum
- 20GB disk space (for models)

### Step 1: Clone and Navigate
```bash
git clone https://github.com/tomblanchard312/SpeechScribe.git
cd SpeechScribe
```

### Step 2: Configure Environment
```bash
cp docker/.env.example docker/.env
# Edit docker/.env with your settings (optional)
```

### Step 3: Start Services
```bash
# Production mode
docker compose -f docker/docker-compose.yml up -d

# Development mode (with hot-reload)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d
```

### Step 4: Access Services
- **Web UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Ollama**: http://localhost:11434

### Step 5: Pull Models
```bash
# Pull a model for Ollama
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5

# Or pull multiple models
docker compose -f docker/docker-compose.yml exec ollama ollama pull llama3
docker compose -f docker/docker-compose.yml exec ollama ollama pull deepseek
```

## Service Architecture

```
┌─────────────────────────────────────────┐
│      Docker Compose Network             │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────┐    ┌──────────────┐   │
│  │ React UI    │    │ FastAPI      │   │
│  │ Port 3000   │    │ Port 8000    │   │
│  │ (Nginx)     ├──→ │ (CUDA)       │   │
│  └─────────────┘    └───────┬──────┘   │
│                              │          │
│                      ┌───────▼──────┐   │
│                      │ Ollama       │   │
│                      │ Port 11434   │   │
│                      │ (CUDA)       │   │
│                      └──────────────┘   │
│                                         │
│  Volumes:                               │
│  • models/    (whisper, ollama)         │
│  • plugins/   (custom plugins)          │
│  • cache/     (model cache)             │
│                                         │
└─────────────────────────────────────────┘
```

## File Structure

```
docker/
├── docker-compose.yml       # Production configuration
├── docker-compose.dev.yml   # Development overrides
├── Dockerfile.api          # API service (FastAPI + CUDA)
├── Dockerfile.ui           # UI service (React + Nginx)
├── nginx.conf              # Nginx reverse proxy config
├── .env.example            # Environment variables template
├── quick-start.sh          # Unix/Linux quick start
├── quick-start.bat         # Windows quick start
├── test.sh                 # Unix/Linux test suite
├── test.bat                # Windows test suite
└── README.md               # Full documentation
```

## Services Configuration

### API Service
- **Image**: nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04
- **Framework**: Python 3.11 + FastAPI
- **Port**: 8000
- **GPU**: NVIDIA CUDA 12.2 (all GPUs)
- **Mounts**:
  - `/app/speechscribe` - Core module
  - `/app/src` - API implementation
  - `/app/models` - Persistent model storage
  - `/app/plugins` - Custom plugins
  - `/app/cache` - Model cache

### UI Service
- **Image**: nginx:alpine
- **Framework**: Node 18 + React 18 + Vite
- **Port**: 3000
- **Server**: Nginx with reverse proxy
- **Build**: Multi-stage (Node builder → Nginx)
- **Features**: Gzip compression, security headers

### Ollama Service
- **Image**: ollama/ollama:latest
- **Port**: 11434
- **GPU**: NVIDIA CUDA (all GPUs)
- **Volume**: `/root/.ollama` - Model cache
- **Health**: HTTP health check every 30s

## Environment Variables

Key variables in `.env`:

```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0          # GPU ID (0 for first, all for all)

# API Configuration
API_PORT=8000
API_HOST=0.0.0.0

# UI Configuration
UI_PORT=3000
REACT_APP_API_URL=http://localhost:8000

# Ollama Configuration
OLLAMA_HOST=http://ollama:11434
OLLAMA_DEFAULT_MODEL=qwen2.5

# Model Configuration
WHISPER_MODEL=base              # tiny, base, small, medium, large-v3
WHISPER_DEVICE=cuda             # cuda, cpu

# Logging
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR

# Storage
MODELS_VOLUME_PATH=./docker/models
PLUGINS_VOLUME_PATH=./docker/plugins
CACHE_VOLUME_PATH=./docker/cache

# Advanced
CLIENT_MAX_BODY_SIZE=100        # Max upload size in MB
API_WORKERS=4                   # FastAPI worker count
```

## Testing

### Automated Testing

**Unix/Linux/macOS:**
```bash
cd docker
chmod +x test.sh
./test.sh
```

**Windows:**
```bash
cd docker
test.bat
```

Tests verify:
- ✓ Docker/Docker Compose installed
- ✓ All containers running
- ✓ Network connectivity (ports 3000, 8000, 11434)
- ✓ GPU access in containers
- ✓ API endpoints working
- ✓ Ollama models available
- ✓ Persistent volumes created
- ✓ No critical errors in logs

### Manual Testing

**Check API:**
```bash
# Health check
curl http://localhost:8000/health

# List plugins
curl http://localhost:8000/plugins

# API documentation
# Browser: http://localhost:8000/docs
```

**Check Ollama:**
```bash
# List models
curl http://localhost:11434/api/tags

# Pull a model
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5

# Test with model
docker compose -f docker/docker-compose.yml exec ollama ollama run qwen2.5 "Hello"
```

**Check GPU Support:**
```bash
# GPU info in API container
docker compose -f docker/docker-compose.yml exec speechscribe-api python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')"

# GPU info in Ollama
docker compose -f docker/docker-compose.yml exec ollama nvidia-smi
```

**Chat Test:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Say hello in 5 words"}
    ],
    "model": "qwen2.5",
    "stream": false
  }'
```

## Development Mode

For development with hot-reload:

```bash
# Start with dev overrides
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

# API at http://localhost:8000 (auto-reload on code changes)
# UI at http://localhost:5173 (Vite dev server with HMR)
# Ollama at http://localhost:11434
```

Features:
- Auto-reload on source changes
- Hot Module Replacement (HMR) for React
- Debug logging
- Direct source code mounts (no rebuild needed)

## Common Commands

```bash
# View all services
docker compose -f docker/docker-compose.yml ps

# View logs
docker compose -f docker/docker-compose.yml logs -f

# View specific service logs
docker compose -f docker/docker-compose.yml logs -f speechscribe-api

# Stop all services
docker compose -f docker/docker-compose.yml down

# Restart services
docker compose -f docker/docker-compose.yml restart

# Rebuild images
docker compose -f docker/docker-compose.yml up -d --build

# Enter container shell
docker compose -f docker/docker-compose.yml exec speechscribe-api bash

# View resource usage
docker stats

# Clean up unused volumes
docker volume prune

# Full cleanup (removes volumes!)
docker compose -f docker/docker-compose.yml down -v
```

## Troubleshooting

### Issue: "GPU not available"
**Solution:**
1. Verify NVIDIA Docker runtime: `docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi`
2. Check GPU in compose config: `docker compose -f docker/docker-compose.yml config | grep -A5 devices`
3. Ensure CUDA_VISIBLE_DEVICES is set in .env

### Issue: "Port already in use"
**Solution:**
Change ports in docker-compose.yml or stop other services:
```bash
# Check what's using ports
netstat -tlnp | grep 8000
netstat -tlnp | grep 3000
netstat -tlnp | grep 11434

# Or change ports in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead of 8000
```

### Issue: "Out of memory"
**Solution:**
1. Use smaller model: `WHISPER_MODEL=tiny`
2. Monitor GPU memory: `docker stats`
3. Increase swap: `docker update --memory 16g container_name`

### Issue: "Ollama models not loading"
**Solution:**
```bash
# Check disk space
docker exec speechscribe-ollama df -h

# Check Ollama logs
docker compose -f docker/docker-compose.yml logs ollama

# Try pulling again
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5
```

### Issue: "API not connecting to Ollama"
**Solution:**
1. Verify Ollama is running: `curl http://localhost:11434/api/version`
2. Check API logs: `docker compose -f docker/docker-compose.yml logs speechscribe-api`
3. Verify network: `docker network ls` (should show `docker_speechscribe-network`)

## Production Deployment

### Checklist
- [ ] Set `NODE_ENV=production` and `LOG_LEVEL=WARN`
- [ ] Use strong API keys if exposed to internet
- [ ] Enable HTTPS with reverse proxy (nginx/traefik)
- [ ] Set resource limits in docker-compose.yml
- [ ] Configure backups for ollama-models volume
- [ ] Monitor with Docker stats/Prometheus
- [ ] Use environment-specific .env files
- [ ] Test health checks and restart policies
- [ ] Set up logging aggregation

### Example Production Override
```bash
docker compose -f docker/docker-compose.yml \
  -f docker/docker-compose.prod.yml \
  up -d
```

## Performance Tips

1. **Model Selection**: Use `tiny` or `base` for speed, `large` for accuracy
2. **GPU**: Ensure CUDA_VISIBLE_DEVICES is set correctly
3. **Memory**: Monitor with `docker stats`
4. **Caching**: Models are cached in volumes, don't re-pull
5. **Concurrency**: Adjust API_WORKERS based on CPU cores

## Getting Help

For issues:
1. Check logs: `docker compose -f docker/docker-compose.yml logs -f`
2. Run tests: `docker/test.sh` (or `test.bat` on Windows)
3. Review troubleshooting section above
4. Check GitHub issues: https://github.com/tomblanchard312/SpeechScribe/issues

## Next Steps

1. ✅ Services running
2. ✅ Pull Ollama models
3. ✅ Open WebUI at http://localhost:3000
4. ✅ Start chatting!

For more information, see:
- [Main README.md](../../README.md)
- [Docker README.md](./README.md)
- [QUICKSTART.md](../../QUICKSTART.md)
