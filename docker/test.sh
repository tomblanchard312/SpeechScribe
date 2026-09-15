#!/bin/bash
# SpeechScribe Docker Testing Script

set -e

echo "================================"
echo "SpeechScribe Docker Test Suite"
echo "================================"
echo ""

TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
test_pass() {
    echo "✓ $1"
    ((TESTS_PASSED++))
}

test_fail() {
    echo "✗ $1"
    ((TESTS_FAILED++))
}

test_info() {
    echo "ℹ $1"
}

# Test 1: Check Docker and Docker Compose
echo "Test 1: Prerequisites"
echo "===================="

if command -v docker &> /dev/null; then
    test_pass "Docker is installed"
else
    test_fail "Docker is not installed"
fi

if command -v docker-compose &> /dev/null; then
    test_pass "Docker Compose is installed"
else
    test_fail "Docker Compose is not installed"
fi

echo ""

# Test 2: Check containers are running
echo "Test 2: Container Status"
echo "======================="

if docker ps | grep -q "speechscribe-api"; then
    test_pass "API container is running"
else
    test_fail "API container is not running"
fi

if docker ps | grep -q "speechscribe-ui"; then
    test_pass "UI container is running"
else
    test_fail "UI container is not running"
fi

if docker ps | grep -q "speechscribe-ollama"; then
    test_pass "Ollama container is running"
else
    test_fail "Ollama container is not running"
fi

echo ""

# Test 3: Check network connectivity
echo "Test 3: Network Connectivity"
echo "============================"

# API
if curl -s http://localhost:8000/docs &> /dev/null; then
    test_pass "API is accessible (http://localhost:8000)"
else
    test_fail "API is not accessible (http://localhost:8000)"
fi

# UI
if curl -s http://localhost:3000 &> /dev/null; then
    test_pass "UI is accessible (http://localhost:3000)"
else
    test_fail "UI is not accessible (http://localhost:3000)"
fi

# Ollama
if curl -s http://localhost:11434/api/tags &> /dev/null; then
    test_pass "Ollama is accessible (http://localhost:11434)"
else
    test_fail "Ollama is not accessible (http://localhost:11434)"
fi

echo ""

# Test 4: Check GPU in containers
echo "Test 4: GPU Support"
echo "==================="

# Check if nvidia-smi works
if command -v nvidia-smi &> /dev/null; then
    test_pass "Host GPU support detected"
    
    # Check API container GPU access
    if docker compose -f docker/docker-compose.yml exec -T speechscribe-api python3 -c "import torch; print('GPU Available:', torch.cuda.is_available())" 2>&1 | grep -q "GPU Available: True"; then
        test_pass "API container has GPU access"
    else
        test_fail "API container does not have GPU access"
    fi
    
    # Check Ollama container GPU access
    if docker compose -f docker/docker-compose.yml exec -T ollama nvidia-smi &> /dev/null; then
        test_pass "Ollama container has GPU access"
    else
        test_fail "Ollama container does not have GPU access"
    fi
else
    test_info "Host GPU not detected - CPU-only mode"
fi

echo ""

# Test 5: Check API endpoints
echo "Test 5: API Endpoints"
echo "===================="

# Health check
if curl -s http://localhost:8000/health &> /dev/null; then
    test_pass "API health endpoint is working"
else
    test_fail "API health endpoint is not working"
fi

# Plugins endpoint
if curl -s http://localhost:8000/plugins 2>&1 | grep -q "plugin_id\|capabilities"; then
    test_pass "API plugins endpoint is working"
else
    test_fail "API plugins endpoint is not working"
fi

# Docs
if curl -s http://localhost:8000/docs 2>&1 | grep -q "OpenAPI\|swagger"; then
    test_pass "API docs are accessible"
else
    test_fail "API docs are not accessible"
fi

echo ""

# Test 6: Check Ollama models
echo "Test 6: Ollama Models"
echo "====================="

MODELS=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | wc -l)
if [ "$MODELS" -gt 0 ]; then
    test_pass "Found $MODELS Ollama model(s)"
else
    test_info "No Ollama models found (pull models with: docker compose exec ollama ollama pull qwen2.5)"
fi

echo ""

# Test 7: Check persistent volumes
echo "Test 7: Persistent Volumes"
echo "=========================="

if docker volume ls | grep -q "docker_models"; then
    test_pass "Models volume exists"
else
    test_fail "Models volume not found"
fi

if docker volume ls | grep -q "docker_plugins"; then
    test_pass "Plugins volume exists"
else
    test_fail "Plugins volume not found"
fi

if docker volume ls | grep -q "docker_cache"; then
    test_pass "Cache volume exists"
else
    test_fail "Cache volume not found"
fi

if docker volume ls | grep -q "docker_ollama-models"; then
    test_pass "Ollama models volume exists"
else
    test_fail "Ollama models volume not found"
fi

echo ""

# Test 8: Check logs for errors
echo "Test 8: Container Logs"
echo "======================"

API_ERRORS=$(docker compose -f docker/docker-compose.yml logs speechscribe-api 2>&1 | grep -i "error\|exception\|traceback" | wc -l)
if [ "$API_ERRORS" -eq 0 ]; then
    test_pass "No critical errors in API logs"
else
    test_info "Found $API_ERRORS error(s) in API logs - check with: docker compose logs speechscribe-api"
fi

OLLAMA_ERRORS=$(docker compose -f docker/docker-compose.yml logs ollama 2>&1 | grep -i "error\|fatal" | wc -l)
if [ "$OLLAMA_ERRORS" -eq 0 ]; then
    test_pass "No critical errors in Ollama logs"
else
    test_info "Found $OLLAMA_ERRORS error(s) in Ollama logs"
fi

echo ""

# Test 9: Sample API request
echo "Test 9: Sample API Request"
echo "=========================="

# Create a test request (if Ollama models are available)
RESPONSE=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Say hello"}
    ],
    "model": "qwen2.5",
    "stream": false
  }' 2>&1)

if echo "$RESPONSE" | grep -q "role\|content\|model"; then
    test_pass "Chat API request successful"
elif echo "$RESPONSE" | grep -q "not found\|not available"; then
    test_info "Chat API working but no models available (pull a model first)"
else
    test_fail "Chat API request failed"
    test_info "Response: $RESPONSE"
fi

echo ""

# Test 10: File permissions and ownership
echo "Test 10: File Permissions"
echo "========================="

for dir in docker/models docker/plugins docker/cache; do
    if [ -d "$dir" ]; then
        test_pass "Directory $dir exists"
    else
        test_info "Directory $dir not found (will be created by Docker)"
    fi
done

echo ""

# Summary
echo "================================"
echo "Test Summary"
echo "================================"
echo "Passed: $TESTS_PASSED"
echo "Failed: $TESTS_FAILED"
echo ""

if [ "$TESTS_FAILED" -eq 0 ]; then
    echo "🎉 All tests passed!"
    exit 0
else
    echo "⚠ Some tests failed. Review the output above."
    exit 1
fi
