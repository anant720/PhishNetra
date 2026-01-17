# PhishNetra - Deployment Guide

This guide covers deployment options for RiskAnalyzer AI in various environments.

## 🚀 Quick Start with Docker

### Prerequisites
- Docker and Docker Compose
- At least 8GB RAM
- 10GB free disk space

### Single-Command Deployment

```bash
# Clone the repository
git clone <repository-url>
cd riskanalyzer-ai

# Start all services
docker-compose -f docker/docker-compose.yml up -d

# Check status
docker-compose -f docker/docker-compose.yml ps

# View logs
docker-compose -f docker/docker-compose.yml logs -f
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## 🏗️ Production Deployment

### 1. Environment Setup

```bash
# Create production environment file
cp backend/env.example backend/.env

# Edit configuration for production
nano backend/.env
```

Key production settings:
```bash
ENVIRONMENT=production
DEBUG_MODE=false
API_WORKERS=4
ENABLE_GPU=true  # If GPU available
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60
```

### 2. SSL/TLS Configuration

#### Using Nginx Reverse Proxy

```bash
# Enable nginx profile
docker-compose -f docker/docker-compose.yml --profile nginx up -d

# Or use certbot for SSL
certbot --nginx -d yourdomain.com
```

#### Direct SSL with FastAPI

```bash
# Install SSL certificates
# Configure uvicorn with SSL
uvicorn main:app --ssl-keyfile key.pem --ssl-certfile cert.pem
```

### 3. Scaling Configuration

#### Horizontal Scaling (Multiple Instances)

```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  backend:
    deploy:
      replicas: 3
    environment:
      - API_WORKERS=2
```

#### Load Balancing

```bash
# Use nginx as load balancer
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.scale.yml up -d
```

## ☁️ Cloud Deployment

### AWS Deployment

#### EC2 Instance

```bash
# Launch EC2 instance (t3.medium or larger)
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Deploy
git clone <repository-url>
cd riskanalyzer-ai
docker-compose up -d
```

#### ECS/Fargate

```yaml
# ecs-task-definition.json
{
  "family": "riskanalyzer-task",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "your-registry/riskanalyzer-backend:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "production"},
        {"name": "API_WORKERS", "value": "2"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/riskanalyzer",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Run

```bash
# Build and push images
gcloud builds submit --tag gcr.io/project-id/riskanalyzer-backend
gcloud builds submit --tag gcr.io/project-id/riskanalyzer-frontend

# Deploy backend
gcloud run deploy riskanalyzer-backend \
  --image gcr.io/project-id/riskanalyzer-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 10 \
  --port 8000

# Deploy frontend
gcloud run deploy riskanalyzer-frontend \
  --image gcr.io/project-id/riskanalyzer-frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1
```

### Azure Container Instances

```bash
# Create resource group
az group create --name riskanalyzer-rg --location eastus

# Create container group
az container create \
  --resource-group riskanalyzer-rg \
  --name riskanalyzer-backend \
  --image your-registry/riskanalyzer-backend:latest \
  --ports 8000 \
  --cpu 1 \
  --memory 2 \
  --environment-variables ENVIRONMENT=production API_WORKERS=2
```

## 📊 Monitoring & Observability

### Basic Monitoring

```bash
# Check container health
docker-compose ps

# View resource usage
docker stats

# Monitor logs
docker-compose logs -f backend
```

### Advanced Monitoring

Enable monitoring stack:
```bash
# Start with monitoring profile
docker-compose --profile monitoring up -d

# Access dashboards
# Grafana: http://localhost:3001 (admin/admin)
# Prometheus: http://localhost:9090
```

### Key Metrics to Monitor

- **API Performance**: Response time, throughput, error rate
- **Model Inference**: Latency, memory usage, GPU utilization
- **System Resources**: CPU, memory, disk I/O
- **Business Metrics**: Analysis requests, risk score distribution

## 🔧 Maintenance

### Model Updates

```bash
# Update models (when new training data available)
docker-compose exec backend python training/scripts/train_distilbert.py

# Reload models without restarting
docker-compose exec backend python -c "from app.models import fasttext_model; fasttext_model._load_model()"
```

### Backup Strategy

```bash
# Backup models and data
docker run --rm -v riskanalyzer_models:/data -v $(pwd)/backup:/backup alpine tar czf /backup/models-$(date +%Y%m%d).tar.gz -C /data .

# Backup database (if used)
docker exec riskanalyzer-postgres pg_dump -U user dbname > backup.sql
```

### Log Management

```bash
# Rotate logs
docker-compose exec backend logrotate /etc/logrotate.d/riskanalyzer

# Archive old logs
docker-compose exec backend find /app/logs -name "*.log" -mtime +30 -delete
```

## 🚨 Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Reduce batch size
MODEL_BATCH_SIZE=8

# Enable model caching
ENABLE_MODEL_CACHING=true

# Monitor memory
docker stats
```

#### Slow Inference
```bash
# Enable GPU if available
ENABLE_GPU=true

# Increase workers
API_WORKERS=4

# Profile performance
python -m cProfile -s time your_script.py
```

#### Connection Refused
```bash
# Check service status
docker-compose ps

# Restart services
docker-compose restart backend frontend

# Check logs
docker-compose logs backend
```

### Performance Tuning

#### FastAPI Optimization
```python
# In main.py
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
```

#### Database Connection Pooling
```python
# Configure connection pool
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
```

## 🔒 Security Checklist

- [ ] SSL/TLS certificates installed
- [ ] Environment variables secured
- [ ] Rate limiting configured
- [ ] CORS properly configured
- [ ] Security headers enabled
- [ ] Regular dependency updates
- [ ] Log monitoring active
- [ ] Backup strategy implemented
- [ ] Access controls configured

## 📞 Support

For deployment issues:
1. Check the logs: `docker-compose logs`
2. Verify configuration: `docker-compose config`
3. Test API endpoints: `curl http://localhost:8000/api/v1/health`
4. Check system resources: `docker stats`

For production support, contact the development team.