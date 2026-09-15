#!/bin/bash
# SpeechScribe Docker Quick Start Script

set -e

echo "================================"
echo "SpeechScribe Docker Quick Start"
echo "================================"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✓ Docker is installed"
echo "✓ Docker Compose is installed"

# Check for GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader
else
    echo "⚠ No NVIDIA GPU detected. CPU-only mode will be used."
fi

echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Create .env if it doesn't exist
if [ ! -f docker/.env ]; then
    echo "Creating .env file from template..."
    cp docker/.env.example docker/.env
    echo "✓ Created docker/.env (please edit if needed)"
fi

echo ""
echo "Starting SpeechScribe services..."
echo ""

# Build and start services
docker compose -f docker/docker-compose.yml up -d

echo ""
echo "================================"
echo "Services Starting..."
echo "================================"
echo ""

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Check API health
echo ""
echo "Checking API health..."
for i in {1..30}; do
    if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
        echo "✓ API is ready at http://localhost:8000"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠ API might not be ready yet. Check logs with: docker compose -f docker/docker-compose.yml logs speechscribe-api"
    fi
    sleep 1
done

# Check UI
echo ""
echo "Checking UI..."
for i in {1..15}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "✓ UI is ready at http://localhost:3000"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "⚠ UI might not be ready yet. Check logs with: docker compose -f docker/docker-compose.yml logs speechscribe-ui"
    fi
    sleep 1
done

# Check Ollama
echo ""
echo "Checking Ollama..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama is ready at http://localhost:11434"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠ Ollama might not be ready yet. Check logs with: docker compose -f docker/docker-compose.yml logs ollama"
    fi
    sleep 1
done

echo ""
echo "================================"
echo "🎉 Deployment Complete!"
echo "================================"
echo ""
echo "Services:"
echo "  Web UI:            http://localhost:3000"
echo "  API Docs:          http://localhost:8000/docs"
echo "  API ReDoc:         http://localhost:8000/redoc"
echo "  Ollama API:        http://localhost:11434"
echo ""
echo "Next steps:"
echo "  1. Open http://localhost:3000 in your browser"
echo "  2. Pull Ollama models:"
echo "     docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5"
echo "  3. Start chatting!"
echo ""
echo "View logs:"
echo "  All services:  docker compose -f docker/docker-compose.yml logs -f"
echo "  API only:      docker compose -f docker/docker-compose.yml logs -f speechscribe-api"
echo "  UI only:       docker compose -f docker/docker-compose.yml logs -f speechscribe-ui"
echo "  Ollama only:   docker compose -f docker/docker-compose.yml logs -f ollama"
echo ""
echo "Stop services:"
echo "  docker compose -f docker/docker-compose.yml down"
echo ""
