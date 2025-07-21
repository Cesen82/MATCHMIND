#!/bin/bash

# MATCHMIND Quick Setup Script
# This script helps you quickly set up the MATCHMIND project

echo "🧠 MATCHMIND Quick Setup"
echo "======================="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
echo "Checking Python version..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [ $(echo "$PYTHON_VERSION >= 3.9" | bc) -eq 1 ]; then
        echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
    else
        echo -e "${RED}✗ Python 3.9+ required (found $PYTHON_VERSION)${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi

# Check Docker
echo "Checking Docker..."
if command_exists docker; then
    echo -e "${GREEN}✓ Docker found${NC}"
else
    echo -e "${YELLOW}⚠ Docker not found (optional but recommended)${NC}"
fi

# Check PostgreSQL client
echo "Checking PostgreSQL client..."
if command_exists psql; then
    echo -e "${GREEN}✓ PostgreSQL client found${NC}"
else
    echo -e "${YELLOW}⚠ PostgreSQL client not found${NC}"
fi

# Create virtual environment
echo
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
echo "Installing dependencies..."
if pip install -r requirements.txt; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi

# Install development dependencies
echo "Installing development dependencies..."
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt > /dev/null 2>&1
    echo -e "${GREEN}✓ Development dependencies installed${NC}"
fi

# Organize repository structure
echo
echo "Organizing repository structure..."
if [ -f "organize_repository.py" ]; then
    python organize_repository.py
else
    echo -e "${YELLOW}⚠ Repository organizer not found${NC}"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env file from template${NC}"
        echo -e "${YELLOW}⚠ Please edit .env with your configuration${NC}"
    fi
else
    echo -e "${YELLOW}⚠ .env file already exists${NC}"
fi

# Create necessary directories
echo
echo "Creating additional directories..."
mkdir -p logs reports models data/raw data/processed
echo -e "${GREEN}✓ Directories created${NC}"

# Database setup prompt
echo
read -p "Do you want to set up the database now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Setting up database..."
    if [ -f "scripts/init_db_script.py" ]; then
        python scripts/init_db_script.py
        echo -e "${GREEN}✓ Database initialized${NC}"
    else
        echo -e "${RED}✗ Database initialization script not found${NC}"
    fi
fi

# Docker setup prompt
echo
read -p "Do you want to build Docker containers? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command_exists docker; then
        echo "Building Docker containers..."
        docker-compose build
        echo -e "${GREEN}✓ Docker containers built${NC}"
    else
        echo -e "${RED}✗ Docker not installed${NC}"
    fi
fi

# Success message
echo
echo -e "${GREEN}✅ Setup complete!${NC}"
echo
echo "Next steps:"
echo "1. Edit .env file with your API keys and configuration"
echo "2. Initialize the database: python scripts/init_db_script.py"
echo "3. Collect initial data: python -m data.data_collector --historical"
echo "4. Train models: python scripts/train_script.py"
echo "5. Start the application:"
echo "   - API Server: python -m api.api_server"
echo "   - Dashboard: streamlit run ui/main.py"
echo "   - Scheduler: python -m services.scheduler"
echo
echo "For more information, see README.md"
echo
echo "Happy predicting! ⚽ 🎯"