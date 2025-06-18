# main.py
"""
Main application entry point for Norwegian Service Aggregator
Production-ready Streamlit application with comprehensive error handling and monitoring
"""

import sys
import os
import logging
import signal
import atexit
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from service_extractor import ServiceExtractor, run_extraction_in_streamlit


# Configure comprehensive logging before any other imports
def setup_logging() -> logging.Logger:
    """Setup comprehensive logging configuration for production environment"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure logging format for production
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    formatter = logging.Formatter(log_format)

    # Create handlers
    app_file_handler = logging.FileHandler(log_dir / 'app.log', encoding='utf-8')
    app_file_handler.setLevel(logging.INFO)
    app_file_handler.setFormatter(formatter)

    error_file_handler = logging.FileHandler(log_dir / 'error.log', encoding='utf-8')
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger("app")
    logger.setLevel(logging.DEBUG)  # Or INFO depending on your verbosity needs
    logger.addHandler(app_file_handler)
    logger.addHandler(error_file_handler)
    logger.addHandler(console_handler)

    # Silence noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    return logger


# Initialize logger
logger = setup_logging()


def setup_environment() -> bool:
    """
    Setup application environment and validate prerequisites
    
    Returns:
        bool: True if environment setup successful, False otherwise
    """
    try:
        # Add project root to Python path for module imports
        project_root = Path(__file__).parent.absolute()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Create required directories
        required_dirs = ['data', 'logs', 'config', 'temp']
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
        
        # Set environment variables for Streamlit optimization
        os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')
        os.environ.setdefault('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
        os.environ.setdefault('STREAMLIT_SERVER_ENABLE_CORS', 'false')
        os.environ.setdefault('STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION', 'true')
        
        # Configure asyncio for Windows compatibility
        import platform
        if platform.system() == 'Windows':
            import asyncio
            if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                logger.info("Set Windows-compatible asyncio event loop policy")
        
        logger.info("Environment setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup environment: {e}", exc_info=True)
        return False


def validate_dependencies() -> bool:
    """
    Validate that all required dependencies are available
    
    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    required_packages = [
        ('streamlit', 'Streamlit web framework'),
        ('pandas', 'Data manipulation library'),
        ('plotly', 'Interactive visualization library'),
        ('requests', 'HTTP client library'),
        ('bs4', 'HTML parsing library'),
        ('playwright', 'Browser automation library')
    ]
    
    missing_packages = []
    
    for package_name, description in required_packages:
        try:
            __import__(package_name)
            logger.debug(f"✅ {package_name}: Available")
        except ImportError:
            missing_packages.append((package_name, description))
            logger.error(f"❌ {package_name}: Missing - {description}")
    
    if missing_packages:
        logger.error("Missing required dependencies:")
        for package_name, description in missing_packages:
            logger.error(f"  - {package_name}: {description}")
        logger.error("Install missing packages with: pip install -r requirements.txt")
        return False
    
    logger.info("All required dependencies are available")
    return True


def setup_error_handlers() -> None:
    """Setup global error handlers for graceful application shutdown"""

    def signal_handler(signum: int, frame) -> None:
        """Handle system signals for graceful shutdown"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        cleanup_application()
        sys.exit(0)

    def exception_handler(exc_type, exc_value, exc_traceback) -> None:
        """Handle uncaught exceptions"""
        if issubclass(exc_type, KeyboardInterrupt):
            logger.info("Application interrupted by user")
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical(
            "Uncaught exception occurred", exc_info=(exc_type, exc_value, exc_traceback)
        )

    import threading

    if threading.current_thread() is threading.main_thread():
        # Register signal handlers safely
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    else:
        logger.warning("Signal handlers not registered: not in main thread")

    # Register exception handler and exit cleanup
    sys.excepthook = exception_handler
    atexit.register(cleanup_application)

    logger.info("Error handlers registered successfully")


def cleanup_application() -> None:
    """Cleanup application resources on shutdown"""
    try:
        logger.info("Starting application cleanup...")
        
        # Cleanup extraction engine if it exists
        try:
            from core.extractor_engine import shutdown_extraction_engine
            shutdown_extraction_engine()
            logger.info("Extraction engine shutdown completed")
        except ImportError:
            logger.debug("Extraction engine not available for cleanup")
        except Exception as e:
            logger.error(f"Error during extraction engine cleanup: {e}")
        
        # Clear Streamlit cache
        try:
            if hasattr(st, 'cache_data'):
                st.cache_data.clear()
            if hasattr(st, 'cache_resource'):
                st.cache_resource.clear()
            logger.info("Streamlit cache cleared")
        except Exception as e:
            logger.error(f"Error clearing Streamlit cache: {e}")
        
        # Close any open file handles
        try:
            import gc
            gc.collect()
            logger.info("Garbage collection completed")
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
        
        logger.info("Application cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during application cleanup: {e}", exc_info=True)


def check_system_resources() -> bool:
    """
    Check system resources and warn if insufficient
    
    Returns:
        bool: True if resources are adequate, False if critical shortage
    """
    try:
        import psutil
        
        # Check available memory
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        
        if available_mb < 512:  # Less than 512 MB available
            logger.error(f"Insufficient memory: {available_mb:.0f} MB available (minimum 512 MB required)")
            return False
        elif available_mb < 1024:  # Less than 1 GB available
            logger.warning(f"Low memory: {available_mb:.0f} MB available (recommended: 1+ GB)")
        else:
            logger.info(f"Memory check passed: {available_mb:.0f} MB available")
        
        # Check disk space
        disk = psutil.disk_usage('.')
        free_mb = disk.free / (1024 * 1024)
        
        if free_mb < 100:  # Less than 100 MB free
            logger.error(f"Insufficient disk space: {free_mb:.0f} MB free (minimum 100 MB required)")
            return False
        elif free_mb < 500:  # Less than 500 MB free
            logger.warning(f"Low disk space: {free_mb:.0f} MB free (recommended: 500+ MB)")
        else:
            logger.info(f"Disk space check passed: {free_mb:.0f} MB free")
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            logger.warning(f"High CPU usage: {cpu_percent:.1f}% (may impact performance)")
        else:
            logger.info(f"CPU usage check passed: {cpu_percent:.1f}%")
        
        return True
        
    except ImportError:
        logger.warning("psutil not available, skipping system resource checks")
        return True
    except Exception as e:
        logger.error(f"Error checking system resources: {e}")
        return True  # Don't fail startup for resource check errors


def initialize_application() -> bool:
    """
    Initialize the main application with comprehensive error handling
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        logger.info("=== Norwegian Service Aggregator v5 Starting ===")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Setup environment and validate prerequisites
        if not setup_environment():
            logger.error("Environment setup failed")
            return False
        
        # Validate dependencies
        if not validate_dependencies():
            logger.error("Dependency validation failed")
            return False
        
        # Check system resources
        if not check_system_resources():
            logger.error("System resource check failed")
            return False
        
        # Setup error handlers
        setup_error_handlers()
        
        logger.info("Application initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Application initialization failed: {e}", exc_info=True)
        return False


def run_application() -> None:
    """Run the main Streamlit application with error handling"""
    try:
        # Import and run the Streamlit application
        from ui.streamlit_app import main as streamlit_main
        
        logger.info("Starting Streamlit application...")
        
        # Run the main application
        streamlit_main()
        
    except ImportError as e:
        error_msg = f"Failed to import required modules: {e}"
        logger.error(error_msg)
        
        # Show error in Streamlit if possible
        try:
            st.error("❌ Application Import Error")
            st.error(error_msg)
            st.info("""
            **Troubleshooting Steps:**
            1. Ensure all dependencies are installed: `pip install -r requirements.txt`
            2. Check that you're in the correct directory
            3. Verify Python path configuration
            4. Check the application logs for detailed error information
            """)
        except:
            print(f"❌ Import Error: {error_msg}")
            print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"Application runtime error: {e}"
        logger.error(error_msg, exc_info=True)
        
        # Show error in Streamlit if possible
        try:
            st.error("❌ Application Runtime Error")
            st.error(error_msg)
            st.info("""
            **Error Recovery Steps:**
            1. Refresh the page to restart the application
            2. Check system resources (memory, disk space)
            3. Review the application logs for detailed information
            4. Contact support if the issue persists
            """)
            
            # Show system information for debugging
            with st.expander("🔧 System Information (for debugging)"):
                st.code(f"""
                Python Version: {sys.version}
                Working Directory: {os.getcwd()}
                Python Path: {sys.path[:3]}...
                Environment Variables: {len(os.environ)} variables set
                """)
                
        except:
            print(f"❌ Application Error: {error_msg}")
            print("Check the logs for detailed error information")
        
        sys.exit(1)


def display_startup_banner() -> None:
    """Display startup banner with system information"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🇳🇴 Norwegian Service Aggregator v5.0.0                ║
    ║                                                              ║
    ║        Real-time Service Plan Extraction & Comparison        ║
    ║                                                              ║
    ║        ⚡ Production-Ready  📊 Analytics  🔒 Secure           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    print(banner)
    logger.info("Norwegian Service Aggregator v5.0.0 - Production Ready")


def main() -> None:
    """
    Main application entry point with comprehensive error handling and monitoring
    Coordinates application initialization, configuration, and execution
    """
    try:
        # Display startup banner
        display_startup_banner()
        
        # Initialize application
        if not initialize_application():
            logger.error("Application initialization failed, exiting...")
            sys.exit(1)
        
        # Run the main application
        run_application()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n👋 Application stopped by user")
        sys.exit(0)
        
    except SystemExit as e:
        logger.info(f"Application exiting with code: {e.code}")
        sys.exit(e.code)
        
    except Exception as e:
        logger.critical(f"Critical application error: {e}", exc_info=True)
        print(f"❌ Critical Error: {e}")
        print("Check logs/app.log for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    # Entry point for direct script execution
    main()