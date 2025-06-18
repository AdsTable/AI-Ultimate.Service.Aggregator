#!/usr/bin/env python3
"""
Deployment script for Norwegian Service Aggregator
Automated deployment with health checks and rollback capabilities
"""

import subprocess
import sys
import time
import requests
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DeploymentManager:
    """
    Comprehensive deployment manager with health checks and rollback capabilities
    Handles Docker Compose deployments with production-ready features
    """
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent
        self.compose_file = self.project_root / "docker-compose.yml"
        self.health_check_timeout = 300  # 5 minutes
        self.health_check_interval = 10  # 10 seconds
        
    def deploy(self, pull_images: bool = True, build_cache: bool = True) -> bool:
        """
        Execute comprehensive deployment with validation and health checks
        
        Args:
            pull_images: Whether to pull latest base images
            build_cache: Whether to use Docker build cache
            
        Returns:
            bool: True if deployment successful, False otherwise
        """
        logger.info(f"Starting deployment for environment: {self.environment}")
        
        try:
            # Pre-deployment validation
            if not self._validate_environment():
                return False
            
            # Backup current state
            if not self._backup_current_state():
                logger.warning("Failed to backup current state, continuing...")
            
            # Pull latest images if requested
            if pull_images:
                if not self._pull_images():
                    return False
            
            # Build application images
            if not self._build_images(use_cache=build_cache):
                return False
            
            # Deploy services
            if not self._deploy_services():
                return False
            
            # Health checks
            if not self._wait_for_health():
                logger.error("Health checks failed, initiating rollback")
                self._rollback()
                return False
            
            # Post-deployment tasks
            if not self._post_deployment_tasks():
                logger.warning("Some post-deployment tasks failed")
            
            logger.info("✅ Deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed with exception: {e}")
            self._rollback()
            return False
    
    def _validate_environment(self) -> bool:
        """Validate deployment environment and prerequisites"""
        logger.info("Validating deployment environment...")
        
        # Check Docker and Docker Compose availability
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.error("Docker or Docker Compose not available")
            return False
        
        # Check if compose file exists
        if not self.compose_file.exists():
            logger.error(f"Docker Compose file not found: {self.compose_file}")
            return False
        
        # Check required directories
        required_dirs = ["data", "logs", "config", "nginx"]
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                logger.info(f"Creating required directory: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Validate environment variables
        required_env_vars = ["POSTGRES_PASSWORD", "REDIS_PASSWORD"]
        missing_vars = []
        
        for var in required_env_vars:
            if not self._get_env_var(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
            logger.info("Using default values for missing variables")
        
        logger.info("✅ Environment validation completed")
        return True
    
    def _get_env_var(self, var_name: str, default: str = None) -> Optional[str]:
        """Get environment variable with optional default"""
        import os
        return os.getenv(var_name, default)
    
    def _backup_current_state(self) -> bool:
        """Backup current deployment state for rollback"""
        logger.info("Creating backup of current state...")
        
        try:
            # Create backup directory with timestamp
            backup_dir = self.project_root / f"backup_{int(time.time())}"
            backup_dir.mkdir(exist_ok=True)
            
            # Backup data directory
            data_dir = self.project_root / "data"
            if data_dir.exists():
                subprocess.run([
                    "cp", "-r", str(data_dir), str(backup_dir / "data")
                ], check=True)
            
            # Backup configuration
            config_dir = self.project_root / "config"
            if config_dir.exists():
                subprocess.run([
                    "cp", "-r", str(config_dir), str(backup_dir / "config")
                ], check=True)
            
            # Export current container states
            result = subprocess.run([
                "docker-compose", "-f", str(self.compose_file), "ps", "--format", "json"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                with open(backup_dir / "container_states.json", "w") as f:
                    f.write(result.stdout)
            
            logger.info(f"✅ Backup created: {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def _pull_images(self) -> bool:
        """Pull latest base images"""
        logger.info("Pulling latest base images...")
        
        try:
            result = subprocess.run([
                "docker-compose", "-f", str(self.compose_file), "pull"
            ], check=True, capture_output=True, text=True)
            
            logger.info("✅ Images pulled successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull images: {e}")
            return False
    
    def _build_images(self, use_cache: bool = True) -> bool:
        """Build application images"""
        logger.info("Building application images...")
        
        build_args = [
            "docker-compose", "-f", str(self.compose_file), "build"
        ]
        
        if not use_cache:
            build_args.append("--no-cache")
        
        try:
            result = subprocess.run(build_args, check=True, capture_output=True, text=True)
            logger.info("✅ Images built successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build images: {e}")
            logger.error(f"Build output: {e.stderr}")
            return False
    
    def _deploy_services(self) -> bool:
        """Deploy services using Docker Compose"""
        logger.info("Deploying services...")
        
        try:
            # Stop existing services gracefully
            subprocess.run([
                "docker-compose", "-f", str(self.compose_file), "down", "--timeout", "30"
            ], check=True)
            
            # Start services
            result = subprocess.run([
                "docker-compose", "-f", str(self.compose_file), "up", "-d", "--remove-orphans"
            ], check=True, capture_output=True, text=True)
            
            logger.info("✅ Services deployed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to deploy services: {e}")
            logger.error(f"Deploy output: {e.stderr}")
            return False
    
    def _wait_for_health(self) -> bool:
        """Wait for all services to become healthy"""
        logger.info("Waiting for services to become healthy...")
        
        services = self._get_service_health_endpoints()
        start_time = time.time()
        
        while time.time() - start_time < self.health_check_timeout:
            all_healthy = True
            
            for service_name, endpoint in services.items():
                if not self._check_service_health(service_name, endpoint):
                    all_healthy = False
                    break
            
            if all_healthy:
                logger.info("✅ All services are healthy")
                return True
            
            logger.info("Waiting for services to become healthy...")
            time.sleep(self.health_check_interval)
        
        logger.error("❌ Health check timeout exceeded")
        return False
    
    def _get_service_health_endpoints(self) -> Dict[str, str]:
        """Get health check endpoints for all services"""
        return {
            "app": "http://localhost:8501/_stcore/health",
            "postgres": "postgresql://aggregator:secure_password@localhost:5432/service_aggregator",
            "redis": "redis://localhost:6379/ping",
            "nginx": "http://localhost:80/health"
        }
    
    def _check_service_health(self, service_name: str, endpoint: str) -> bool:
        """Check health of individual service"""
        try:
            if endpoint.startswith("http"):
                response = requests.get(endpoint, timeout=5)
                healthy = response.status_code == 200
            elif endpoint.startswith("postgresql"):
                # PostgreSQL health check
                import psycopg2
                conn = psycopg2.connect(endpoint)
                conn.close()
                healthy = True
            elif endpoint.startswith("redis"):
                # Redis health check
                import redis
                r = redis.Redis.from_url("redis://localhost:6379")
                healthy = r.ping()
            else:
                healthy = False
            
            if healthy:
                logger.debug(f"✅ {service_name} is healthy")
            else:
                logger.debug(f"❌ {service_name} is not healthy")
            
            return healthy
            
        except Exception as e:
            logger.debug(f"❌ {service_name} health check failed: {e}")
            return False
    
    def _post_deployment_tasks(self) -> bool:
        """Execute post-deployment tasks"""
        logger.info("Executing post-deployment tasks...")
        
        success = True
        
        # Database migrations
        if not self._run_database_migrations():
            logger.error("Database migrations failed")
            success = False
        
        # Clear caches
        if not self._clear_caches():
            logger.warning("Cache clearing failed")
        
        # Update monitoring configuration
        if not self._update_monitoring():
            logger.warning("Monitoring update failed")
        
        # Send deployment notification
        self._send_deployment_notification(success)
        
        return success
    
    def _run_database_migrations(self) -> bool:
        """Run database migrations"""
        logger.info("Running database migrations...")
        
        try:
            # Run migrations inside the app container
            result = subprocess.run([
                "docker-compose", "-f", str(self.compose_file), 
                "exec", "-T", "app", "python", "-c", 
                "from core.database import run_migrations; run_migrations()"
            ], check=True, capture_output=True, text=True)
            
            logger.info("✅ Database migrations completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Database migrations failed: {e}")
            return False
    
    def _clear_caches(self) -> bool:
        """Clear application caches"""
        logger.info("Clearing caches...")
        
        try:
            # Clear Redis cache
            subprocess.run([
                "docker-compose", "-f", str(self.compose_file), 
                "exec", "-T", "redis", "redis-cli", "FLUSHALL"
            ], check=True)
            
            logger.info("✅ Caches cleared")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Cache clearing failed: {e}")
            return False
    
    def _update_monitoring(self) -> bool:
        """Update monitoring configuration"""
        logger.info("Updating monitoring configuration...")
        
        try:
            # Reload Prometheus configuration
            subprocess.run([
                "docker-compose", "-f", str(self.compose_file), 
                "exec", "-T", "prometheus", "kill", "-HUP", "1"
            ], check=True)
            
            logger.info("✅ Monitoring configuration updated")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Monitoring update failed: {e}")
            return False
    
    def _send_deployment_notification(self, success: bool):
        """Send deployment notification to monitoring systems"""
        status = "SUCCESS" if success else "FAILED"
        message = f"Deployment {status} for Norwegian Service Aggregator ({self.environment})"
        
        logger.info(f"📢 {message}")
        
        # Here you could integrate with Slack, Teams, email, etc.
        # Example: send to webhook
        # webhook_url = self._get_env_var("DEPLOYMENT_WEBHOOK_URL")
        # if webhook_url:
        #     requests.post(webhook_url, json={"text": message})
    
    def _rollback(self) -> bool:
        """Rollback to previous deployment state"""
        logger.info("🔄 Initiating rollback to previous state...")
        
        try:
            # Find most recent backup
            backup_dirs = [d for d in self.project_root.glob("backup_*") if d.is_dir()]
            if not backup_dirs:
                logger.error("No backup found for rollback")
                return False
            
            latest_backup = max(backup_dirs, key=lambda d: d.stat().st_mtime)
            logger.info(f"Rolling back to: {latest_backup}")
            
            # Stop current services
            subprocess.run([
                "docker-compose", "-f", str(self.compose_file), "down", "--timeout", "30"
            ])
            
            # Restore data
            data_backup = latest_backup / "data"
            if data_backup.exists():
                subprocess.run([
                    "cp", "-r", str(data_backup), str(self.project_root / "data")
                ], check=True)
            
            # Restore configuration
            config_backup = latest_backup / "config"
            if config_backup.exists():
                subprocess.run([
                    "cp", "-r", str(config_backup), str(self.project_root / "config")
                ], check=True)
            
            # Start services with previous configuration
            subprocess.run([
                "docker-compose", "-f", str(self.compose_file), "up", "-d"
            ], check=True)
            
            logger.info("✅ Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def status(self) -> Dict[str, any]:
        """Get current deployment status"""
        try:
            result = subprocess.run([
                "docker-compose", "-f", str(self.compose_file), "ps", "--format", "json"
            ], capture_output=True, text=True, check=True)
            
            services = json.loads(result.stdout) if result.stdout else []
            
            status_info = {
                "environment": self.environment,
                "services": services,
                "healthy_services": 0,
                "total_services": len(services)
            }
            
            # Check health of each service
            health_endpoints = self._get_service_health_endpoints()
            for service in services:
                service_name = service.get("Service", "")
                endpoint = health_endpoints.get(service_name)
                
                if endpoint and self._check_service_health(service_name, endpoint):
                    status_info["healthy_services"] += 1
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {"error": str(e)}


def main():
    """Main deployment script entry point"""
    parser = argparse.ArgumentParser(description="Norwegian Service Aggregator Deployment Script")
    parser.add_argument("--environment", "-e", default="production", 
                       help="Deployment environment")
    parser.add_argument("--no-pull", action="store_true", 
                       help="Skip pulling latest base images")
    parser.add_argument("--no-cache", action="store_true", 
                       help="Build without cache")
    parser.add_argument("--status", action="store_true", 
                       help="Show deployment status")
    parser.add_argument("--rollback", action="store_true", 
                       help="Rollback to previous deployment")
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    deployer = DeploymentManager(environment=args.environment)
    
    # Execute requested operation
    if args.status:
        status = deployer.status()
        print(json.dumps(status, indent=2))
        return
    
    if args.rollback:
        success = deployer._rollback()
        sys.exit(0 if success else 1)
    
    # Execute deployment
    success = deployer.deploy(
        pull_images=not args.no_pull,
        build_cache=not args.no_cache
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()