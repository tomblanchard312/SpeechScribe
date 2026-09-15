# Docker Deployment Verification Checklist

## Pre-Deployment Checklist

### System Requirements
- [ ] Docker 20.10+ installed
- [ ] Docker Compose 1.29+ installed  
- [ ] NVIDIA Docker runtime installed (for GPU)
- [ ] 8GB RAM available
- [ ] 20GB disk space available
- [ ] NVIDIA GPU available (optional, CPU mode supported)

### Configuration
- [ ] `.env` file created from `.env.example`
- [ ] `CUDA_VISIBLE_DEVICES` set correctly
- [ ] `REACT_APP_API_URL` matches your deployment
- [ ] `OLLAMA_HOST` configured correctly
- [ ] Volume paths accessible

### Code & Repository
- [ ] Repository cloned
- [ ] Branch checked out (main/develop)
- [ ] No uncommitted changes
- [ ] Python dependencies updated
- [ ] Node dependencies installed (`npm ci` in ui directory)

## Deployment Steps

### 1. Build Services
- [ ] API image builds successfully
  ```bash
  docker build -f docker/Dockerfile.api -t speechscribe-api:v1.0 .
  ```
- [ ] UI image builds successfully
  ```bash
  docker build -f docker/Dockerfile.ui -t speechscribe-ui:v1.0 .
  ```
- [ ] No build errors or warnings
- [ ] Image sizes reasonable (< 3GB for API, < 200MB for UI)

### 2. Start Services
- [ ] All services start without errors
  ```bash
  docker compose -f docker/docker-compose.yml up -d
  ```
- [ ] No immediate exit codes
- [ ] Health checks initial within expected time

### 3. Verify Service Health (30 seconds after startup)

#### API Service
- [ ] Container is running
  ```bash
  docker ps | grep speechscribe-api
  ```
- [ ] Port 8000 is open
  ```bash
  curl http://localhost:8000/health
  ```
- [ ] API documentation accessible
  ```bash
  curl http://localhost:8000/docs
  ```
- [ ] Plugins endpoint working
  ```bash
  curl http://localhost:8000/plugins
  ```
- [ ] No critical errors in logs
  ```bash
  docker logs speechscribe-api | grep -i "error\|exception"
  ```

#### UI Service
- [ ] Container is running
  ```bash
  docker ps | grep speechscribe-ui
  ```
- [ ] Port 3000 is accessible
  ```bash
  curl http://localhost:3000
  ```
- [ ] Nginx proxy working
  ```bash
  curl -I http://localhost:3000/api/plugins
  ```
- [ ] No critical errors in logs
  ```bash
  docker logs speechscribe-ui | grep -i "error"
  ```

#### Ollama Service
- [ ] Container is running
  ```bash
  docker ps | grep ollama
  ```
- [ ] Port 11434 is responding
  ```bash
  curl http://localhost:11434/api/version
  ```
- [ ] Health check passing
  ```bash
  docker ps --format "table {{.Names}}\t{{.Status}}" | grep ollama
  ```
- [ ] No critical errors in logs
  ```bash
  docker logs speechscribe-ollama | grep -i "error\|fatal"
  ```

#### Network
- [ ] Services can reach each other
  ```bash
  docker compose -f docker/docker-compose.yml exec speechscribe-api curl http://ollama:11434/api/version
  ```
- [ ] Custom bridge network created
  ```bash
  docker network ls | grep speechscribe
  ```

## GPU Verification (if applicable)

- [ ] NVIDIA Docker runtime available
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
  ```
- [ ] API container has GPU access
  ```bash
  docker compose -f docker/docker-compose.yml exec speechscribe-api python3 -c "import torch; print(torch.cuda.is_available())"
  ```
- [ ] Ollama using GPU
  ```bash
  docker compose -f docker/docker-compose.yml exec ollama nvidia-smi
  ```
- [ ] CUDA cores available
  ```bash
  docker compose -f docker/docker-compose.yml exec speechscribe-api nvidia-smi
  ```

## Volume Verification

- [ ] Models volume created
  ```bash
  docker volume ls | grep models
  ```
- [ ] Plugins volume created
  ```bash
  docker volume ls | grep plugins
  ```
- [ ] Cache volume created
  ```bash
  docker volume ls | grep cache
  ```
- [ ] Ollama models volume created
  ```bash
  docker volume ls | grep ollama
  ```
- [ ] Volumes are writable
  ```bash
  docker compose -f docker/docker-compose.yml exec speechscribe-api touch /app/models/test.txt
  ```

## Ollama Model Setup

- [ ] At least one model pulled
  ```bash
  docker compose -f docker/docker-compose.yml exec ollama ollama list
  ```
- [ ] Default model is available
  ```bash
  docker compose -f docker/docker-compose.yml exec ollama ollama list | grep qwen2.5
  ```
- [ ] Models accessible from API
  ```bash
  curl http://localhost:8000/plugins | jq '.[] | select(.type=="summarization")'
  ```

## Functional Testing

### API Tests
- [ ] Health endpoint responds
  ```bash
  curl http://localhost:8000/health
  ```
- [ ] Plugins list returns all plugins
  ```bash
  curl http://localhost:8000/plugins | jq 'length'
  ```
- [ ] Chat endpoint accepts requests
  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"test"}],"model":"qwen2.5"}'
  ```
- [ ] API returns valid responses
  ```bash
  # Check response has required fields
  curl http://localhost:8000/chat | jq 'keys'
  ```

### UI Tests
- [ ] UI loads without 404s
  ```bash
  curl -I http://localhost:3000 | grep 200
  ```
- [ ] React bundles load
  ```bash
  curl http://localhost:3000 | grep "script\|link" | wc -l
  ```
- [ ] API calls from UI work
  ```bash
  # Check browser console in UI for API call success
  ```

### Integration Tests
- [ ] UI can fetch plugin list
  ```bash
  curl http://localhost:3000/api/plugins | jq '.[] | .name'
  ```
- [ ] UI can send chat messages
  ```bash
  curl -X POST http://localhost:3000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello"}],"model":"qwen2.5"}'
  ```

## Performance Checks

- [ ] CPU usage reasonable (<80%)
  ```bash
  docker stats --no-stream
  ```
- [ ] Memory usage acceptable (<6GB for API)
  ```bash
  docker stats --no-stream
  ```
- [ ] No major memory leaks over 5 minutes
  ```bash
  docker stats --no-stream
  ```
- [ ] Response times acceptable (<2 seconds for API)
  ```bash
  time curl http://localhost:8000/plugins
  ```
- [ ] GPU memory not exhausted
  ```bash
  docker compose -f docker/docker-compose.yml exec ollama nvidia-smi
  ```

## Logging & Monitoring

- [ ] Logs are being generated
  ```bash
  docker compose -f docker/docker-compose.yml logs --tail=20
  ```
- [ ] No continuous error loops
  ```bash
  docker compose -f docker/docker-compose.yml logs | grep -c "error"
  ```
- [ ] Log format is readable
  ```bash
  docker compose -f docker/docker-compose.yml logs speechscribe-api | head -5
  ```
- [ ] Timestampsd are present
  ```bash
  docker compose -f docker/docker-compose.yml logs --timestamps
  ```

## Security Checks

- [ ] No hardcoded credentials in images
  ```bash
  docker run speechscribe-api:v1.0 env | grep -i "pass\|key\|secret\|token"
  ```
- [ ] API doesn't expose sensitive endpoints
  ```bash
  curl http://localhost:8000/openapi.json | grep -i "admin\|secret"
  ```
- [ ] CORS properly configured
  ```bash
  curl -H "Origin: http://other-site.com" -I http://localhost:8000/plugins
  ```
- [ ] No default credentials in use
  ```bash
  # Verify .env doesn't contain defaults
  ```

## Restart & Recovery

- [ ] Services restart on failure
  ```bash
  docker ps -a | grep "always\|unless-stopped"
  ```
- [ ] Health checks trigger restarts
  ```bash
  docker compose -f docker/docker-compose.yml exec speechscribe-pi curl http://localhost:8000/health
  ```
- [ ] Data persists after restart
  ```bash
  # Note current model count, restart, verify count unchanged
  ```
- [ ] Clean shutdown without data loss
  ```bash
  docker compose -f docker/docker-compose.yml down
  docker compose -f docker/docker-compose.yml up -d
  ```

## Post-Deployment

### Documentation
- [ ] README updated with deployment info
- [ ] Environment variables documented
- [ ] Troubleshooting guide tested
- [ ] Quick start scripts verified

### Monitoring Setup
- [ ] Logging configured
- [ ] Alerts configured (if applicable)
- [ ] Backup strategy documented
- [ ] Update process documented

### Team Communication
- [ ] Deployment documented
- [ ] Access URLs shared
- [ ] Known issues listed
- [ ] Support contacts provided

## Rollback Plan

- [ ] Previous version tagged and available
  ```bash
  docker images | grep speechscribe
  ```
- [ ] Rollback process documented
- [ ] Volume backups recent
  ```bash
  ls -lh docker/models
  ```
- [ ] Tested rollback once (recommended)

## Sign Off

- [ ] Deployment Lead: _________________ Date: _______
- [ ] DevOps: _________________ Date: _______
- [ ] QA: _________________ Date: _______

## Notes

```
[Space for deployment notes]
```
