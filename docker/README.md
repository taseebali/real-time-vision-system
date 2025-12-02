# Docker Deployment Guide

Complete guide for running Real Time Vision System in Docker containers with GPU support.

## Prerequisites

### 1. Docker Installation

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
```

### 2. NVIDIA Docker Runtime (for GPU support)

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 3. Docker Compose (optional but recommended)

```bash
sudo apt-get install docker-compose-plugin
```

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Navigate to project root
cd /path/to/real-time-vision-system

# Set your camera IP
export CAMERA_IP=192.168.1.100

# Build and run
docker-compose -f docker/docker-compose.yml up --build

# Run in background
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop
docker-compose -f docker/docker-compose.yml down
```

### Using Docker Directly

```bash
# Build the image
docker build -t real-time-vision-system:latest -f docker/Dockerfile .

# Run with GPU support
docker run --rm -it \
  --gpus all \
  --network host \
  -e DISPLAY=$DISPLAY \
  -e CAMERA_IP=192.168.1.100 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/models:/app/models \
  real-time-vision-system:latest

# Run in headless mode (no display)
docker run --rm -it \
  --gpus all \
  --network host \
  -e CAMERA_IP=192.168.1.100 \
  -e SHOW_DISPLAY=false \
  -v $(pwd)/models:/app/models \
  real-time-vision-system:latest
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_IP` | `192.168.1.100` | IP Webcam address |
| `MODEL_NAME` | `yolov8m.pt` | YOLOv8 model (n/s/m/l/x) |
| `CONFIDENCE_THRESHOLD` | `0.30` | Detection confidence |
| `SHOW_DISPLAY` | `true` | Show visual display |
| `DISPLAY` | `:0` | X11 display for GUI |

### Example with Custom Configuration

```bash
docker run --rm -it \
  --gpus all \
  --network host \
  -e CAMERA_IP=192.168.1.150 \
  -e MODEL_NAME=yolov8n.pt \
  -e CONFIDENCE_THRESHOLD=0.40 \
  -e SHOW_DISPLAY=true \
  -v $(pwd)/models:/app/models \
  real-time-vision-system:latest
```

## Building Images

### Standard Build

```bash
# Build with default settings
docker build -t real-time-vision-system:latest -f docker/Dockerfile .

# Build with specific tag
docker build -t real-time-vision-system:v0.1.0-beta -f docker/Dockerfile .
```

### Multi-stage Build (CPU-only)

For CPU-only deployment (no GPU):

```dockerfile
# Modify Dockerfile first line to:
FROM python:3.11-slim

# Then remove CUDA installation and use CPU PyTorch:
RUN pip install torch torchvision torchaudio
```

### Build Arguments

```bash
# Build with custom Python version
docker build \
  --build-arg PYTHON_VERSION=3.11 \
  -t real-time-vision-system:latest \
  -f docker/Dockerfile .
```

## Display Setup (X11 Forwarding)

### Linux

```bash
# Allow X11 connections
xhost +local:docker

# Run container
docker-compose -f docker/docker-compose.yml up

# Revoke access when done
xhost -local:docker
```

### Windows (WSL2)

```bash
# Install VcXsrv or Xming on Windows

# In WSL2, set DISPLAY variable
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0

# Allow connections in Windows X server
# Add to config: -ac (disable access control)

# Run container
docker-compose -f docker/docker-compose.yml up
```

### macOS

```bash
# Install XQuartz
brew install --cask xquartz

# Start XQuartz and allow network connections
# XQuartz → Preferences → Security → "Allow connections from network clients"

# Set DISPLAY
export DISPLAY=host.docker.internal:0

# Allow connections
xhost + 127.0.0.1

# Run container
docker-compose -f docker/docker-compose.yml up
```

## Volume Mounts

### Important Directories

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./src` | `/app/src` | Source code (read-only) |
| `./models` | `/app/models` | Model cache (persistent) |
| `./tests` | `/app/tests` | Test data (read-only) |
| `./test_output` | `/app/test_output` | Output files |
| `/tmp/.X11-unix` | `/tmp/.X11-unix` | X11 socket for display |

### Custom Volume Example

```yaml
volumes:
  # Mount custom model directory
  - /path/to/custom/models:/app/models
  
  # Mount recordings directory
  - ./recordings:/app/recordings
```

## Networking

### Host Network Mode (Default)

Used for easy camera access and X11 display:

```yaml
network_mode: host
```

### Bridge Network Mode (Alternative)

For better isolation:

```yaml
networks:
  - real-time-vision-system-net

networks:
  real-time-vision-system-net:
    driver: bridge
```

## GPU Configuration

### Check GPU Access

```bash
# Inside container
docker exec -it real-time-vision-system nvidia-smi

# Or run directly
docker run --rm --gpus all real-time-vision-system:latest nvidia-smi
```

### Limit GPU Usage

```bash
# Use specific GPU
docker run --gpus '"device=0"' ...

# Limit GPU memory
docker run --gpus all \
  --env CUDA_VISIBLE_DEVICES=0 \
  --memory=8g \
  ...
```

## Troubleshooting

### No GPU Detected

```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check docker daemon configuration
cat /etc/docker/daemon.json

# Should contain:
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

### Display Not Working

```bash
# Check X11 socket permissions
ls -la /tmp/.X11-unix/

# Allow X11 connections
xhost +local:docker

# Check DISPLAY variable
echo $DISPLAY

# Test with simple GUI app
docker run --rm -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  ubuntu:22.04 \
  sh -c "apt-get update && apt-get install -y x11-apps && xeyes"
```

### Camera Connection Issues

```bash
# Check network connectivity
docker exec -it real-time-vision-system ping 192.168.1.100

# Test camera URL
docker exec -it real-time-vision-system curl http://192.168.1.100:8080/video

# Use host network mode
network_mode: host
```

### Out of Memory

```bash
# Increase shared memory
--shm-size=2g

# Or in docker-compose.yml:
shm_size: '2gb'

# Use smaller model
-e MODEL_NAME=yolov8n.pt
```

### Permission Denied

```bash
# Run as current user
docker run --user $(id -u):$(id -g) ...

# Or fix permissions
sudo chown -R $(id -u):$(id -g) models/ test_output/
```

## Production Deployment

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker/docker-compose.yml real-time-vision-system

# Check status
docker stack services real-time-vision-system

# Remove stack
docker stack rm real-time-vision-system
```

### Using Kubernetes

Create `real-time-vision-system-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: real-time-vision-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: real-time-vision-system
  template:
    metadata:
      labels:
        app: real-time-vision-system
    spec:
      containers:
      - name: real-time-vision-system
        image: real-time-vision-system:latest
        env:
        - name: CAMERA_IP
          value: "192.168.1.100"
        resources:
          limits:
            nvidia.com/gpu: 1
```

Deploy:

```bash
kubectl apply -f real-time-vision-system-deployment.yaml
```

## Performance Optimization

### Build Cache

```bash
# Use BuildKit for better caching
DOCKER_BUILDKIT=1 docker build -t real-time-vision-system:latest -f docker/Dockerfile .
```

### Multi-stage Build

```dockerfile
# Builder stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as builder
RUN pip install --user torch torchvision torchaudio

# Runtime stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
COPY --from=builder /root/.local /root/.local
```

### Image Size Reduction

```bash
# Check image size
docker images real-time-vision-system

# Remove unnecessary files
# Add to Dockerfile:
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && pip cache purge
```

## Security Best Practices

### Non-root User

```dockerfile
# Add to Dockerfile
RUN useradd -m -u 1000 assistant
USER assistant
```

### Read-only Filesystem

```yaml
# In docker-compose.yml
security_opt:
  - no-new-privileges:true
read_only: true
tmpfs:
  - /tmp
```

### Secrets Management

```bash
# Use Docker secrets
docker secret create camera_ip camera_ip.txt

# In compose:
secrets:
  - camera_ip
```

## Maintenance

### Update Dependencies

```bash
# Rebuild with latest packages
docker-compose -f docker/docker-compose.yml build --no-cache

# Update base image
docker pull nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
```

### Clean Up

```bash
# Remove unused images
docker image prune -a

# Remove stopped containers
docker container prune

# Remove unused volumes
docker volume prune

# Clean everything
docker system prune -a --volumes
```

## Monitoring

### Resource Usage

```bash
# Monitor container resources
docker stats real-time-vision-system

# Check logs
docker logs -f real-time-vision-system

# Execute commands inside container
docker exec -it real-time-vision-system bash
```

### Health Checks

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' real-time-vision-system

# View health check logs
docker inspect --format='{{json .State.Health}}' real-time-vision-system | jq
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Docker Build

on:
  push:
    branches: [main, dev]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t real-time-vision-system:${{ github.sha }} -f docker/Dockerfile .
    
    - name: Test image
      run: |
        docker run --rm real-time-vision-system:${{ github.sha }} python -m pytest tests/
```

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

## Support

For issues related to:
- Docker setup: Check [Docker docs](https://docs.docker.com/)
- GPU access: Check [NVIDIA Container Toolkit docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Application issues: Open an issue on GitHub

---

**Note**: This deployment is optimized for development. For production, consider additional security hardening, monitoring, and orchestration solutions.
