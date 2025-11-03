#!/bin/bash

# Mental Health MLOps - Setup Script
# This script initializes the project environment

set -e

echo "🚀 Setting up Mental Health MLOps Pipeline..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
python3 --version

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
python3 -m venv venv

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
pip install -r requirements.txt

# Install package in editable mode
echo -e "${BLUE}Installing package in editable mode...${NC}"
pip install -e .

# Initialize Git
if [ ! -d .git ]; then
    echo -e "${BLUE}Initializing Git repository...${NC}"
    git init
    git add .
    git commit -m "Initial commit: Project structure setup"
else
    echo -e "${GREEN}Git repository already initialized${NC}"
fi

# Initialize DVC
if [ ! -d .dvc ]; then
    echo -e "${BLUE}Initializing DVC...${NC}"
    dvc init
    git add .dvc .dvcignore
    git commit -m "Initialize DVC"
else
    echo -e "${GREEN}DVC already initialized${NC}"
fi

# Install pre-commit hooks
echo -e "${BLUE}Installing pre-commit hooks...${NC}"
pre-commit install

# Create .env file from template
if [ ! -f .env ]; then
    echo -e "${BLUE}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${GREEN}Please update .env file with your credentials${NC}"
else
    echo -e "${GREEN}.env file already exists${NC}"
fi

# Create logs directory
mkdir -p logs

echo -e "${GREEN}✅ Setup complete!${NC}"
echo -e "${GREEN}Next steps:${NC}"
echo "1. Update .env file with your AWS credentials"
echo "2. Place your dataset in data/raw/ directory"
echo "3. Run: dvc add data/raw/mental_health_data.csv"
echo "4. Run: dvc remote add -d myremote s3://your-bucket-name"
echo "5. Run: dvc push"
echo ""
echo "To activate the virtual environment, run: source venv/bin/activate"