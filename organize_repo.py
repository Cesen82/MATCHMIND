"""
Repository Organization Script for MATCHMIND
===========================================

This script organizes all project files into the correct directory structure.
"""

import os
import shutil
from pathlib import Path

# Define the directory structure
DIRECTORY_STRUCTURE = {
    'api': ['api_server.py'],
    'core': [
        'predictor_model.py',
        'bet_optimizer.py',
        'feature_engineering.py',
        'backtesting.py'
    ],
    'data': [
        'data_collector.py',
        'database_manager.py',
        'cache_manager.py',
        'football_api.py'
    ],
    'ui': ['main.py'],
    'ui/pages': [
        'dashboard_page.py',
        'predictions_page.py',
        'analytics_page.py',
        'betting_slips_page.py'
    ],
    'ui/components': [
        'charts_module.py',
        'ui_theme.py'
    ],
    'services': [
        'scheduler.py',
        'notification_service.py',
        'model_evaluator.py'
    ],
    'utils': [
        'config.py',
        'logger_module.py',
        'validators_module.py',
        'formatters_module.py',
        'exceptions_module.py',
        'rate_limiter.py'
    ],
    'docker': [],
    'tests': [
        'test_models.py',
        'test_utils.py'
    ],
    'docs': [],
    'scripts': [
        'init_db_script.py',
        'train_script.py',
        'setup.py'
    ],
    'reports': [],
    'models': [],
    'logs': []
}

# Files that should be in root directory
ROOT_FILES = [
    'README.md',
    'requirements.txt',
    'requirements-dev.txt',
    'pyproject.toml',
    'Makefile',
    '.gitignore',
    '.env.example',
    'docker-compose.yml',
    'Dockerfile'
]


def create_directory_structure():
    """Create all necessary directories."""
    print("Creating directory structure...")
    
    for directory in DIRECTORY_STRUCTURE.keys():
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}/")
        
        # Create __init__.py files for Python packages
        if not directory in ['docker', 'docs', 'reports', 'models', 'logs']:
            init_file = Path(directory) / '__init__.py'
            if not init_file.exists():
                init_file.write_text('"""Package initialization."""\n')


def move_files():
    """Move files to their correct locations."""
    print("\nOrganizing files...")
    
    # Get all Python files in current directory
    current_files = list(Path('.').glob('*.py'))
    
    moved_count = 0
    for file_path in current_files:
        filename = file_path.name
        
        # Skip this script
        if filename == 'organize_repository.py':
            continue
            
        # Find correct directory for file
        moved = False
        for directory, files in DIRECTORY_STRUCTURE.items():
            if filename in files:
                target_dir = Path(directory)
                target_path = target_dir / filename
                
                # Move file
                try:
                    shutil.move(str(file_path), str(target_path))
                    print(f"✓ Moved {filename} -> {directory}/")
                    moved = True
                    moved_count += 1
                    break
                except Exception as e:
                    print(f"✗ Error moving {filename}: {e}")
                    
        if not moved and filename not in ROOT_FILES:
            print(f"? Unknown file: {filename} (left in root)")
            
    print(f"\nMoved {moved_count} files")


def create_env_example():
    """Create .env.example file."""
    env_content = """# MATCHMIND Environment Configuration
# Copy this file to .env and fill in your values

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/matchmind
REDIS_URL=redis://localhost:6379/0

# API Keys
FOOTBALL_API_KEY=your_api_football_key_here
ODDS_API_KEY=your_odds_api_key_here

# Security
SECRET_KEY=your-secret-key-here-change-this
API_USERNAME=admin
API_PASSWORD=change_this_password

# Application Settings
APP_ENV=development
DEBUG=True
LOG_LEVEL=INFO

# Email Configuration (for notifications)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=noreply@matchmind.com

# Telegram Bot (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_IDS=123456789,987654321

# Discord Webhook (optional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Webhook Security
WEBHOOK_SECRET=your_webhook_secret_here

# Model Settings
MODEL_UPDATE_FREQUENCY=daily
CONFIDENCE_THRESHOLD=0.65

# Betting Settings
MAX_STAKE_PERCENTAGE=0.1
MIN_STAKE=5.0
COMMISSION=0.05

# Supported Leagues (comma-separated)
SUPPORTED_LEAGUES=premier-league,la-liga,serie-a,bundesliga,ligue-1

# Data Retention
DATA_RETENTION_DAYS=365

# Timezone
TIMEZONE=Europe/Rome
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_content)
    print("✓ Created .env.example")


def create_docker_files():
    """Create Docker configuration files."""
    # Main Dockerfile
    dockerfile_content = """FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 matchmind && chown -R matchmind:matchmind /app
USER matchmind

# Expose ports
EXPOSE 8000 8501

# Default command
CMD ["python", "-m", "uvicorn", "api.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    with open('docker/Dockerfile', 'w') as f:
        f.write(dockerfile_content)
    print("✓ Created docker/Dockerfile")
    
    # Move docker-compose.yml to root if it exists
    if Path('docker-compose.yml').exists():
        print("✓ docker-compose.yml already in root")
    elif Path('docker-compose.txt').exists():
        shutil.move('docker-compose.txt', 'docker-compose.yml')
        print("✓ Renamed docker-compose.txt to docker-compose.yml")


def create_additional_docs():
    """Create additional documentation files."""
    # API documentation
    api_doc = """# MATCHMIND API Documentation

## Authentication
All API endpoints require HTTP Basic Authentication.

### Headers
```
Authorization: Basic base64(username:password)
```

## Endpoints

### Predictions

#### Get Today's Predictions
```
GET /predict/today
```

Parameters:
- `league` (optional): Filter by league
- `min_confidence` (optional): Minimum confidence threshold (0.0-1.0)

#### Get Match Prediction
```
POST /predict/match
```

Body:
```json
{
    "match_id": "string",
    "include_live_data": true
}
```

### Betting

#### Optimize Betting Strategy
```
POST /betting/optimize
```

Body:
```json
{
    "capital": 1000,
    "strategy": "kelly",
    "risk_level": "medium",
    "leagues": ["premier-league"]
}
```

### Data

#### Get Leagues
```
GET /data/leagues
```

#### Get Teams
```
GET /data/teams/{league_id}
```

## Response Format

All responses follow this format:
```json
{
    "data": {},
    "status": "success",
    "timestamp": "2024-01-01T00:00:00Z"
}
```

Error responses:
```json
{
    "error": "Error message",
    "status_code": 400,
    "timestamp": "2024-01-01T00:00:00Z"
}
```
"""
    
    with open('docs/API.md', 'w') as f:
        f.write(api_doc)
    print("✓ Created docs/API.md")
    
    # Deployment guide
    deployment_doc = """# MATCHMIND Deployment Guide

## Prerequisites
- Docker and Docker Compose installed
- PostgreSQL database
- Redis server
- Domain name (optional)
- SSL certificate (recommended)

## Quick Deployment

### 1. Clone and Configure
```bash
git clone https://github.com/Cesen82/MATCHMIND.git
cd MATCHMIND
cp .env.example .env
# Edit .env with your configuration
```

### 2. Build and Start Services
```bash
docker-compose up -d
```

### 3. Initialize Database
```bash
docker-compose exec app python scripts/init_db_script.py
```

### 4. Train Initial Models
```bash
docker-compose exec app python scripts/train_script.py
```

## Production Deployment

### AWS EC2
1. Launch EC2 instance (t3.large recommended)
2. Install Docker and Docker Compose
3. Configure security groups (ports 80, 443, 8000, 8501)
4. Set up Nginx reverse proxy
5. Configure SSL with Let's Encrypt

### Google Cloud Platform
1. Create GCE instance
2. Use Cloud SQL for PostgreSQL
3. Use Memorystore for Redis
4. Deploy with Cloud Run or GKE

### Monitoring
- Set up Prometheus + Grafana
- Configure alerts for:
  - High CPU/Memory usage
  - Failed predictions
  - API errors
  - Low accuracy

## Security Checklist
- [ ] Change default passwords
- [ ] Enable firewall
- [ ] Set up SSL/TLS
- [ ] Rotate API keys regularly
- [ ] Enable audit logging
- [ ] Regular backups
"""
    
    with open('docs/DEPLOYMENT.md', 'w') as f:
        f.write(deployment_doc)
    print("✓ Created docs/DEPLOYMENT.md")
    
    # Contributing guide
    contributing_doc = """# Contributing to MATCHMIND

## Getting Started
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Code Style
- Follow PEP 8
- Use type hints
- Add docstrings to all functions/classes
- Maximum line length: 100 characters

## Testing
- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## Pull Request Process
1. Update documentation
2. Add tests for new functionality
3. Update README.md if needed
4. Request review from maintainers
"""
    
    with open('docs/CONTRIBUTING.md', 'w') as f:
        f.write(contributing_doc)
    print("✓ Created docs/CONTRIBUTING.md")


def create_gitignore():
    """Create comprehensive .gitignore file."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Environment
.env
.env.local
.env.*.local

# Logs
logs/
*.log

# Database
*.db
*.sqlite3

# Models
models/*.pkl
models/*.joblib
models/*.h5

# Reports
reports/*.html
reports/*.pdf
reports/*.png

# Testing
.coverage
.pytest_cache/
htmlcov/
.tox/

# Distribution
build/
dist/
*.egg-info/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Secrets
secrets/
*.pem
*.key

# Temporary files
tmp/
temp/
*.tmp

# Data files
data/*.csv
data/*.json
data/*.xlsx

# Cache
.cache/
*.cache
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✓ Created .gitignore")


def main():
    """Main function to organize the repository."""
    print("🧠 MATCHMIND Repository Organizer")
    print("=================================\n")
    
    # Create directory structure
    create_directory_structure()
    
    # Move files to correct locations
    move_files()
    
    # Create configuration files
    print("\nCreating configuration files...")
    create_env_example()
    create_docker_files()
    create_additional_docs()
    create_gitignore()
    
    print("\n✅ Repository organization complete!")
    print("\nNext steps:")
    print("1. Review the file organization")
    print("2. Copy .env.example to .env and configure")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Initialize database: python scripts/init_db_script.py")
    print("5. Start developing! 🚀")
    
    # Cleanup
    if Path('organize_repository.py').exists():
        response = input("\nDelete this organization script? (y/n): ")
        if response.lower() == 'y':
            os.remove('organize_repository.py')
            print("✓ Organization script removed")


if __name__ == "__main__":
    main()