#!/usr/bin/env python3
"""
Project Setup Script for Norwegian Service Providers System
Automatically creates the recommended directory structure with Windows compatibility
"""

import os
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# Configure logging for setup process with Windows-compatible format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('setup.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class ProjectStructureCreator:
    """
    Creates and manages project directory structure
    Ensures consistent organization across development environments
    Windows and Unix compatible implementation
    """
    
    def __init__(self, project_root: str = "norwegian-service-providers"):
        """
        Initialize project structure creator with cross-platform support
        
        Args:
            project_root: Root directory name for the project
        """
        self.project_root = Path(project_root)
        
        # Define directory structure - organized by functional areas
        self.directories = [
            # Core application directories
            "config",
            "src",
            "src/core", 
            "src/data",
            "src/scrapers",
            "src/api",
            "src/utils",
            
            # Database and migration directories
            "database",
            "database/migrations",
            "database/seeds",
            
            # Testing infrastructure
            "tests",
            "tests/fixtures",
            "tests/unit",
            "tests/integration",
            
            # Utility and script directories
            "scripts",
            "logs",
            
            # Data storage directories with clear separation
            "data",
            "data/raw",
            "data/processed", 
            "data/cache",
            "data/exports",
            
            # Documentation and deployment
            "docs",
            "deployment"
        ]
        
        # Define files to create with their content
        self.files_to_create = [
            # Python package initialization files
            ("src/__init__.py", self._get_package_init("Norwegian Service Providers System")),
            ("src/core/__init__.py", self._get_package_init("Core business logic modules")),
            ("src/data/__init__.py", self._get_package_init("Data management modules")),
            ("src/scrapers/__init__.py", self._get_package_init("Web scraping modules")),
            ("src/api/__init__.py", self._get_package_init("API modules")),
            ("src/utils/__init__.py", self._get_package_init("Utility modules")),
            ("tests/__init__.py", self._get_package_init("Test modules")),
            ("config/__init__.py", self._get_package_init("Configuration modules")),
            
            # Essential project files
            ("logs/.gitkeep", "# Keep logs directory in git repository"),
            ("data/cache/.gitkeep", "# Keep cache directory structure"),
            (".gitignore", self._get_gitignore_content()),
            ("requirements.txt", self._get_requirements_content()),
            ("requirements-dev.txt", self._get_dev_requirements_content()),
            (".env.example", self._get_env_example_content()),
            ("README.md", self._get_readme_content()),
            ("setup.py", self._get_setup_py_content()),
            ("pyproject.toml", self._get_pyproject_content()),
        ]
    
    def create_structure(self) -> None:
        """
        Create the complete project structure with comprehensive error handling
        Provides detailed feedback and rollback capabilities
        """
        try:
            logger.info(f"Starting project setup for: {self.project_root.absolute()}")
            logger.info(f"Platform: {sys.platform}")
            
            # Validate environment before proceeding
            self._validate_environment()
            
            # Create project root directory
            self.project_root.mkdir(exist_ok=True)
            logger.info(f"Project root created: {self.project_root.absolute()}")
            
            # Create all subdirectories with progress tracking
            self._create_directories()
            
            # Create essential files with content validation
            self._create_files()
            
            # Validate created structure
            self._validate_structure()
            
            logger.info("✅ Project structure created successfully!")
            self._print_next_steps()
            
        except KeyboardInterrupt:
            logger.warning("Setup cancelled by user")
            self._cleanup_partial_setup()
        except Exception as e:
            logger.error(f"Failed to create project structure: {e}")
            self._cleanup_partial_setup()
            raise
    
    def _validate_environment(self) -> None:
        """Validate that the environment is suitable for setup"""
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")
        
        # Check available disk space (basic check)
        try:
            import shutil
            free_space = shutil.disk_usage('.').free
            if free_space < 100 * 1024 * 1024:  # 100MB minimum
                logger.warning("Low disk space detected")
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
        
        logger.info(f"Environment validation passed - Python {sys.version}")
    
    def _create_directories(self) -> None:
        """Create all required directories with progress tracking"""
        total_dirs = len(self.directories)
        
        for idx, directory in enumerate(self.directories, 1):
            dir_path = self.project_root / directory
            
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"[{idx}/{total_dirs}] Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {e}")
                raise
        
        logger.info(f"✅ Created {total_dirs} directories successfully")
    
    def _create_files(self) -> None:
        """Create essential files with content validation"""
        total_files = len(self.files_to_create)
        created_count = 0
        skipped_count = 0
        
        for idx, (file_path, content) in enumerate(self.files_to_create, 1):
            full_path = self.project_root / file_path
            
            try:
                # Skip existing files to prevent overwriting
                if full_path.exists():
                    logger.warning(f"[{idx}/{total_files}] File exists, skipping: {full_path}")
                    skipped_count += 1
                    continue
                
                # Ensure parent directory exists
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write file with proper encoding
                full_path.write_text(content, encoding='utf-8')
                logger.debug(f"[{idx}/{total_files}] Created file: {full_path}")
                created_count += 1
                
            except Exception as e:
                logger.error(f"Failed to create file {full_path}: {e}")
                raise
        
        logger.info(f"✅ Created {created_count} files, skipped {skipped_count} existing files")
    
    def _validate_structure(self) -> None:
        """Validate that the created structure is correct"""
        validation_errors = []
        
        # Check that all directories exist
        for directory in self.directories:
            dir_path = self.project_root / directory
            if not dir_path.exists():
                validation_errors.append(f"Missing directory: {dir_path}")
        
        # Check that essential files exist
        essential_files = ["README.md", "requirements.txt", ".gitignore"]
        for file_name in essential_files:
            file_path = self.project_root / file_name
            if not file_path.exists():
                validation_errors.append(f"Missing essential file: {file_path}")
        
        if validation_errors:
            logger.error("Structure validation failed:")
            for error in validation_errors:
                logger.error(f"  - {error}")
            raise RuntimeError("Project structure validation failed")
        
        logger.info("✅ Project structure validation passed")
    
    def _cleanup_partial_setup(self) -> None:
        """Clean up partially created structure on failure"""
        try:
            if self.project_root.exists():
                logger.info("Cleaning up partial setup...")
                # Note: Only remove if directory is empty or contains only our files
                # This prevents accidental deletion of existing work
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def _get_package_init(self, description: str) -> str:
        """Generate __init__.py content with proper docstring"""
        return f'"""{description}"""\n\n__version__ = "0.1.0"\n'
    
    def _get_gitignore_content(self) -> str:
        """Generate comprehensive .gitignore content for Python projects"""
        content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDEs and editors
.vscode/
.idea/
*.swp
*.swo
*~

# Environment variables
.env
.env.local
.env.production

# Logs
logs/*.log
*.log
setup.log

# Database files
*.db
*.sqlite
*.sqlite3

# Cache directories
data/cache/*
!data/cache/.gitkeep
__pycache__/

# Scraped data (uncomment if you don't want to commit scraped data)
# data/raw/*
# data/processed/*

# Operating System files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Windows specific
*.exe
*.msi
*.msm
*.msp

# Temporary files
*.tmp
*.temp
*.bak
"""
        return content
    
    def _get_requirements_content(self) -> str:
        """Generate requirements.txt with essential dependencies"""
        content = """# Core web scraping and data processing
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.3
selenium>=4.15.0
pandas>=2.1.0
numpy>=1.24.0

# Database handling
# sqlite3 is included in Python standard library

# Web framework options
flask>=2.3.0
fastapi>=0.104.0
uvicorn>=0.24.0

# Environment and configuration management
python-dotenv>=1.0.0
pydantic>=2.4.0
pydantic-settings>=2.0.0

# Logging and monitoring
structlog>=23.2.0
rich>=13.6.0

# Data validation and serialization
marshmallow>=3.20.0
jsonschema>=4.19.0

# HTTP client enhancements
httpx>=0.25.0
aiohttp>=3.8.0

# Utility libraries
click>=8.1.0
tqdm>=4.66.0
python-dateutil>=2.8.0
"""
        return content
    
    def _get_dev_requirements_content(self) -> str:
        """Generate development requirements"""
        content = """# Testing frameworks
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.12.0
pytest-asyncio>=0.21.0

# Code quality and formatting
black>=23.9.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.6.0
pylint>=3.0.0

# Development tools
pre-commit>=3.5.0
jupyter>=1.0.0
ipython>=8.16.0

# Documentation
sphinx>=7.2.0
sphinx-rtd-theme>=1.3.0

# Debugging
debugpy>=1.8.0
pdb-attach>=3.2.0
"""
        return content
    
    def _get_env_example_content(self) -> str:
        """Generate .env.example with comprehensive configuration"""
        content = """# Database Configuration
DATABASE_PATH=database/service_plans.db
DATABASE_BACKUP_PATH=database/backups/

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/application.log
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# Web Scraping Configuration
SCRAPING_DELAY=1
SCRAPING_TIMEOUT=30
SCRAPING_RETRIES=3
USER_AGENT=Mozilla/5.0 (Norwegian Service Provider Bot 1.0)

# Cache Configuration
CACHE_TIMEOUT=300
STATS_CACHE_TIMEOUT=300
CACHE_DIRECTORY=data/cache/

# API Configuration
API_HOST=localhost
API_PORT=8000
API_DEBUG=False
API_WORKERS=1

# Provider URLs (backup configuration)
TELIA_URL=https://www.telia.no/mobilabonnement
TELENOR_URL=https://www.telenor.no/mobilabonnement/
ICE_URL=https://www.ice.no/mobilabonnement/
TALKMORE_URL=https://talkmore.no/privat/abonnement/enkelt/bestill
FORTUM_URL=https://www.fortum.com/no/strom/stromavtale
FORBRUKERRADET_URL=https://www.forbrukerradet.no/strompris/
LYSE_URL=https://www.lyse.no/strom

# Feature Flags
ENABLE_SELENIUM=True
ENABLE_API_INTEGRATION=True
ENABLE_CACHING=True
ENABLE_ANALYTICS=True

# Development Settings
DEBUG=False
TESTING=False
"""
        return content
    
    def _get_readme_content(self) -> str:
        """Generate comprehensive README.md content"""
        content = """# Norwegian Service Providers System

A comprehensive system for tracking and analyzing mobile and electricity service providers in Norway.

## 🚀 Features

- **Provider Data Management**: Track mobile and electricity providers with current pricing
- **Web Scraping**: Automated data extraction from provider websites using Selenium and BeautifulSoup
- **Analytics Dashboard**: Comprehensive analytics and reporting capabilities
- **Data Quality**: Built-in validation and quality scoring mechanisms
- **Intelligent Caching**: Performance optimization through smart caching strategies
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 📋 Requirements

- Python 3.8 or higher
- 100MB free disk space
- Internet connection for provider data fetching

## 🛠️ Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to project directory
cd norwegian-service-providers

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development