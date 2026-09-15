# SpeechScribe Docker Deployment

Complete Docker deployment stack for SpeechScribe with GPU acceleration, Ollama LLM integration, and persistent storage.

## Architecture

```
┌─────────────────────────────────────────────┐
│         Docker Compose Network              │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐    ┌──────────────┐      │
│  │  speechscribe │    │  speechscribe│      │
│  │      ui      │    │     api      │      │
│  │ (Port 3000)  │    │ (Port 8000)  │      │
│  └──────────────┘    └──────────────┘      │
│      (Nginx)          (FastAPI+CUDA)       │
│                              │              │
│                              ▼              │
│                       ┌──────────────┐      │
│                       │    Ollama    │      │
│                       │ (Port 11434) │      │
│                       │   (CUDA)     │      │
│                       └──────────────┘      │
│                                             │
│  Persistent Volumes:                        │
│  ├── models/       (Whisper, Ollama models)│
│  ├── plugins/      (Plugin directory)       │
│  ├── cache/        (Model cache)            │
│  └── ollama-models/(Ollama model storage)  │
│                                             │
└─────────────────────────────────────────────┘
```

## Services

### speechscribe-api
- **Base Image**: nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04
- **Framework**: FastAPI + Uvicorn
- **GPU**: NVIDIA CUDA 12.2 with GPU support
- **Port**: 8000
- **Features**:
  - Multi-stage build for smaller image size
  - Python 3.11 virtual environment
  - All speech processing dependencies (whisper, faster-whisper, torch, librosa)
  - Health checks
  - Auto-restart on failure

### speechscribe-ui
- **Base Image**: node:18-alpine → nginx:alpine
- **Framework**: React 18 + Vite
- **Server**: Nginx with reverse proxy to API
- **Port**: 3000 (exposed as 3000)
- **Features**:
  - Multi-stage build (Node → Nginx)
  - Production build with optimization
  - Gzip compression
  - API proxy with WebSocket support
  - SPA routing fallback
  - Security headers

### ollama
- **Image**: ollama/ollama:latest
- **GPU**: NVIDIA CUDA support
- **Port**: 11434
- **Storage**: Persistent volume for models
- **Features**:
  - Health checks
  - Auto-restart on failure
  - Large model cache support

## Prerequisites

### System Requirements
- **Docker** 20.10+
- **Docker Compose** 1.29+
- **NVIDIA Docker Runtime** (for GPU support)
- **GPU Memory**: 4GB minimum (8GB+ recommended)
- **RAM**: 8GB minimum
- **Disk Space**: 20GB+ (for models)

### Install NVIDIA Docker Runtime

**Ubuntu/Debian:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Windows/macOS with Docker Desktop:**
- Docker Desktop GPU support is available in Docker Desktop 4.3+
- Go to Settings → Resources → GPU and enable GPU support

### Verify GPU Support
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
```

## Quick Start

### 1. Clone and Setup
```bash
cd SpeechScribe
cp docker/.env.example docker/.env
```

### 2. Build and Start Services
```bash
docker compose -f docker/docker-compose.yml up -d
```

Or with specific service:
```bash
docker compose -f docker/docker-compose.yml up -d speechscribe-api
docker compose -f docker/docker-compose.yml up -d speechscribe-ui
docker compose -f docker/docker-compose.yml up -d ollama
```

### 3. Monitor Services
```bash
docker compose -f docker/docker-compose.yml logs -f

# Or specific service
docker compose -f docker/docker-compose.yml logs -f speechscribe-api
```

### 4. Initial Setup

**Pull Ollama Models:**
```bash
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5
docker compose -f docker/docker-compose.yml exec ollama ollama pull llama3
```

## Testing

### Access Services

**Web UI:**
```
http://localhost:3000
```

**API Documentation:**
```
http://localhost:8000/docs
http://localhost:8000/redoc
```

**Ollama API:**
```bash
curl http://localhost:11434/api/tags
```

### Test API Health

```bash
# Check API health
curl http://localhost:8000/health

# List plugins
curl http://localhost:8000/plugins

# Chat with Ollama
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "model": "qwen2.5"
  }'
```

### Test Ollama Models

```bash
# List available models
docker compose -f docker/docker-compose.yml exec ollama ollama list

# Pull additional models
docker compose -f docker/docker-compose.yml exec ollama ollama pull mistral

# Test model
docker compose -f docker/docker-compose.yml exec ollama ollama run qwen2.5 "Hello"
```

### Test GPU Acceleration

```bash
# Check GPU in API container
docker compose -f docker/docker-compose.yml exec speechscribe-api python3 -c "import torch; print('GPU Available:', torch.cuda.is_available())"

# Check CUDA in Ollama
docker compose -f docker/docker-compose.yml exec ollama nvidia-smi
```

## Advanced Usage

### Environment Variables

Copy and edit `.env` file:
```bash
cp docker/.env.example docker/.env
```

Available variables:
- `CUDA_VISIBLE_DEVICES` - GPU ID to use (default: 0)
- `REACT_APP_API_URL` - API endpoint for UI
- `OLLAMA_HOST` - Ollama service URL
- `WHISPER_MODEL` - Whisper model size (tiny, base, small, medium, large)

### Scale Services

```bash
# Run with custom build context
docker compose -f docker/docker-compose.yml build --no-cache

# Push to registry
docker tag speechscribe-api:latest myregistry/speechscribe-api:latest
docker push myregistry/speechscribe-api:latest
```

### View Logs

```bash
# All services
docker compose -f docker/docker-compose.yml logs

# Specific service
docker compose -f docker/docker-compose.yml logs speechscribe-api

# Follow logs
docker compose -f docker/docker-compose.yml logs -f speechscribe-ui

# Last 50 lines
docker compose -f docker/docker-compose.yml logs --tail=50
```

### Manage Services

```bash
# Stop all services
docker compose -f docker/docker-compose.yml down

# Stop and remove volumes
docker compose -f docker/docker-compose.yml down -v

# Restart services
docker compose -f docker/docker-compose.yml restart

# Rebuild images
docker compose -f docker/docker-compose.yml up -d --build

# Enter container shell
docker compose -f docker/docker-compose.yml exec speechscribe-api bash
```

### Persistent Storage

Models and plugins are stored in Docker volumes:

```bash
# List volumes
docker volume ls | grep speechscribe

# Inspect volume
docker volume inspect docker_models

# Backup models
docker run --rm -v docker_models:/data -v $(pwd):/backup \
  alpine tar czf /backup/models-backup.tar.gz -C /data .

# Restore models
docker run --rm -v docker_models:/data -v $(pwd):/backup \
  alpine tar xzf /backup/models-backup.tar.gz -C /data
```

## Troubleshooting

### GPU Not Available in Container

```bash
# Check host GPU
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi

# Check compose GPU config
docker compose -f docker/docker-compose.yml config | grep -A5 devices
```

### Out of Memory

Reduce model size or increase GPU memory:
```bash
# Use smaller whisper model
WHISPER_MODEL=tiny

# Monitor GPU memory
docker compose -f docker/docker-compose.yml exec speechscribe-api nvidia-smi
```

### Ollama Models Not Loading

```bash
# Check Ollama status
docker compose -f docker/docker-compose.yml exec ollama ollama list

# Check available disk space
docker exec speechscribe-ollama df -h

# Pull model with progress
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5
```

### API Container Won't Start

```bash
# Check logs
docker compose -f docker/docker-compose.yml logs speechscribe-api

# Build with debug output
docker compose -f docker/docker-compose.yml build --verbose speechscribe-api

# Run with interactive shell
docker run -it --gpus all speechscribe-api bash
```

### UI Not Connecting to API

1. Check API is running: `docker compose -f docker/docker-compose.yml ps`
2. Test API directly: `curl http://localhost:8000/docs`
3. Check Nginx logs: `docker compose -f docker/docker-compose.yml logs speechscribe-ui`
4. Verify proxy config: Check `nginx.conf` location blocks

## Performance Tuning

### GPU Memory
```yaml
# In docker-compose.yml, adjust:
environment:
  - CUDA_VISIBLE_DEVICES=0
  - CUDA_LAUNCH_BLOCKING=0
```

### CPU and Memory Limits
```yaml
# Limit API service
speechscribe-api:
  deploy:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
```

### Network Optimization
```yaml
# Use host network (Linux only)
network_mode: host
```

## Production Deployment

### Use Docker Registry

```bash
# Build and push
docker build -t myregistry/speechscribe-api:v1.0 -f docker/Dockerfile.api .
docker push myregistry/speechscribe-api:v1.0

# Update docker-compose to use image
speechscribe-api:
  image: myregistry/speechscribe-api:v1.0
```

### Health Checks

Each service includes health checks:
```bash
docker compose -f docker/docker-compose.yml ps
```

### Monitoring

```bash
# Resource usage
docker stats

# Container inspection
docker inspect speechscribe-api

# Event monitoring
docker events --filter type=container
```

## File Structure

```
docker/
├── docker-compose.yml    # Service orchestration
├── Dockerfile.api        # API service image
├── Dockerfile.ui         # UI service image
├── nginx.conf           # Nginx configuration
├── .env.example         # Environment template
└── README.md            # This file

Volumes created:
├── models/              # Whisper, SpeechT5 models
├── plugins/             # Plugin directory
├── cache/               # Model cache
└── ollama-models/       # Ollama LLM models
```

## Security Notes

- Change default ports in production
- Use `.env` file for sensitive configuration
- Enable HTTPS with reverse proxy (nginx, etc.)
- Restrict API endpoints with authentication
- Use private Docker registry for images
- Keep images updated regularly
- Monitor resource usage and logs
- Set resource limits on containers
- Use read-only volumes where possible

## Support

For issues:
1. Check logs: `docker compose logs speechscribe-api`
2. Verify GPU: `nvidia-smi`
3. Test connectivity: `docker exec speechscribe-api curl http://ollama:11434/api/tags`
4. Check volumes: `docker volume ls`
5. Review docker-compose.yml configuration

## License

SpeechScribe is licensed under the MIT License.
