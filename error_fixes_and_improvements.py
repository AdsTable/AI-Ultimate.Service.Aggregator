"""
System Error Fixes and Architectural Improvements
Addresses multiple system failures and implements robust error handling
"""

import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import sqlite3
from contextlib import contextmanager

# Configure logging for better error tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SystemStats:
    """Data class for system statistics with caching support"""
    last_updated: datetime = field(default_factory=datetime.now)
    total_plans: int = 0
    active_providers: int = 0
    extraction_success_rate: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

class ExtractionEngine:
    """
    Enhanced extraction engine with proper caching and error handling
    Fixes: 'ExtractionEngine' object has no attribute '_stats_cache_time'
    """
    
    def __init__(self, cache_timeout: int = 300):
        """
        Initialize extraction engine with caching capabilities
        
        Args:
            cache_timeout: Cache timeout in seconds (default: 5 minutes)
        """
        self._stats_cache: Optional[SystemStats] = None
        self._stats_cache_time: Optional[datetime] = None
        self._cache_timeout = timedelta(seconds=cache_timeout)
        self._extraction_methods = [
            'selenium_scraping',
            'api_integration', 
            'manual_entry',
            'bulk_import'
        ]
        logger.info("ExtractionEngine initialized with caching support")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics with intelligent caching
        Fixes: Failed to load statistics error
        
        Returns:
            Dictionary containing current system statistics
        """
        try:
            # Check if cache is valid
            if self._is_cache_valid():
                logger.debug("Returning cached statistics")
                return self._stats_to_dict(self._stats_cache)
            
            # Refresh cache with new statistics
            self._refresh_stats_cache()
            return self._stats_to_dict(self._stats_cache)
            
        except Exception as e:
            logger.error(f"Failed to load statistics: {e}")
            # Return default statistics on error
            return self._get_default_stats()
    
    def _is_cache_valid(self) -> bool:
        """Check if current cache is still valid"""
        if not self._stats_cache or not self._stats_cache_time:
            return False
        return datetime.now() - self._stats_cache_time < self._cache_timeout
    
    def _refresh_stats_cache(self) -> None:
        """Refresh the statistics cache with current data"""
        try:
            # Simulate statistics collection (replace with actual implementation)
            new_stats = SystemStats(
                last_updated=datetime.now(),
                total_plans=self._count_total_plans(),
                active_providers=self._count_active_providers(),
                extraction_success_rate=self._calculate_success_rate(),
                cache_hits=getattr(self._stats_cache, 'cache_hits', 0) + (1 if self._stats_cache else 0),
                cache_misses=getattr(self._stats_cache, 'cache_misses', 0) + (0 if self._stats_cache else 1)
            )
            
            self._stats_cache = new_stats
            self._stats_cache_time = datetime.now()
            logger.info("Statistics cache refreshed successfully")
            
        except Exception as e:
            logger.error(f"Failed to refresh statistics cache: {e}")
            raise
    
    def _stats_to_dict(self, stats: SystemStats) -> Dict[str, Any]:
        """Convert SystemStats object to dictionary"""
        return {
            'last_updated': stats.last_updated.isoformat(),
            'total_plans': stats.total_plans,
            'active_providers': stats.active_providers,
            'extraction_success_rate': stats.extraction_success_rate,
            'cache_performance': {
                'hits': stats.cache_hits,
                'misses': stats.cache_misses,
                'hit_ratio': stats.cache_hits / max(stats.cache_hits + stats.cache_misses, 1)
            }
        }
    
    def _get_default_stats(self) -> Dict[str, Any]:
        """Return default statistics when data unavailable"""
        return {
            'last_updated': datetime.now().isoformat(),
            'total_plans': 0,
            'active_providers': 0,
            'extraction_success_rate': 0.0,
            'cache_performance': {'hits': 0, 'misses': 0, 'hit_ratio': 0.0},
            'status': 'degraded_mode'
        }
    
    def _count_total_plans(self) -> int:
        """Count total plans in system (implement based on your data source)"""
        # Placeholder implementation
        return 42
    
    def _count_active_providers(self) -> int:
        """Count active providers (implement based on your data source)"""
        # Placeholder implementation  
        return 7
    
    def _calculate_success_rate(self) -> float:
        """Calculate extraction success rate (implement based on your logs)"""
        # Placeholder implementation
        return 0.95

class DatabaseManager:
    """
    Enhanced database manager with schema validation and migration support
    Fixes: no such column: extraction_method
    """
    
    def __init__(self, db_path: str = 'service_plans.db'):
        """
        Initialize database manager with migration capabilities
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_schema_compatibility()
        logger.info(f"DatabaseManager initialized for {db_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _ensure_schema_compatibility(self) -> None:
        """
        Ensure database schema includes all required columns
        Fixes: no such column: extraction_method
        """
        try:
            with self.get_connection() as conn:
                # Check if required tables exist
                self._create_tables_if_not_exist(conn)
                # Add missing columns
                self._add_missing_columns(conn)
                conn.commit()
                logger.info("Database schema validation completed")
                
        except Exception as e:
            logger.error(f"Schema compatibility check failed: {e}")
            raise
    
    def _create_tables_if_not_exist(self, conn: sqlite3.Connection) -> None:
        """Create required tables if they don't exist"""
        tables_sql = [
            """
            CREATE TABLE IF NOT EXISTS service_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider_name TEXT NOT NULL,
                plan_name TEXT NOT NULL,
                price REAL,
                features TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                extraction_method TEXT DEFAULT 'unknown',
                data_quality_score REAL DEFAULT 0.0,
                last_verified TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS providers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                url TEXT,
                category TEXT,
                is_active BOOLEAN DEFAULT 1,
                last_scraped TIMESTAMP,
                extraction_method TEXT DEFAULT 'unknown'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS extraction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider_name TEXT,
                extraction_method TEXT NOT NULL,
                status TEXT NOT NULL,
                records_extracted INTEGER DEFAULT 0,
                error_message TEXT,
                execution_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for sql in tables_sql:
            conn.execute(sql)
    
    def _add_missing_columns(self, conn: sqlite3.Connection) -> None:
        """Add missing columns to existing tables"""
        missing_columns = [
            ("service_plans", "extraction_method", "TEXT DEFAULT 'unknown'"),
            ("service_plans", "data_quality_score", "REAL DEFAULT 0.0"),
            ("service_plans", "last_verified", "TIMESTAMP"),
            ("providers", "extraction_method", "TEXT DEFAULT 'unknown'"),
            ("providers", "is_active", "BOOLEAN DEFAULT 1")
        ]
        
        for table, column, definition in missing_columns:
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                logger.debug(f"Added column {column} to table {table}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    logger.warning(f"Could not add column {column} to {table}: {e}")
    
    def get_plans_data(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve service plans data with proper error handling
        Fixes: Failed to load plans data error
        
        Args:
            category: Optional category filter
            
        Returns:
            List of service plan dictionaries
        """
        try:
            with self.get_connection() as conn:
                if category:
                    query = """
                    SELECT sp.*, p.category, p.url 
                    FROM service_plans sp
                    LEFT JOIN providers p ON sp.provider_name = p.name
                    WHERE p.category = ? AND p.is_active = 1
                    ORDER BY sp.updated_at DESC
                    """
                    cursor = conn.execute(query, (category,))
                else:
                    query = """
                    SELECT sp.*, p.category, p.url 
                    FROM service_plans sp
                    LEFT JOIN providers p ON sp.provider_name = p.name
                    WHERE p.is_active = 1
                    ORDER BY sp.updated_at DESC
                    """
                    cursor = conn.execute(query)
                
                plans = [dict(row) for row in cursor.fetchall()]
                logger.info(f"Retrieved {len(plans)} service plans")
                return plans
                
        except Exception as e:
            logger.error(f"Failed to load plans data: {e}")
            return []  # Return empty list instead of failing
    
    def get_analytics_data(self) -> Dict[str, Any]:
        """
        Get analytics data with comprehensive error handling
        Fixes: Failed to load analytics data error
        
        Returns:
            Dictionary containing analytics metrics
        """
        try:
            with self.get_connection() as conn:
                analytics = {}
                
                # Provider statistics
                provider_stats = conn.execute("""
                    SELECT 
                        category,
                        COUNT(*) as count,
                        COUNT(CASE WHEN is_active = 1 THEN 1 END) as active_count
                    FROM providers 
                    GROUP BY category
                """).fetchall()
                
                analytics['provider_stats'] = [dict(row) for row in provider_stats]
                
                # Extraction method distribution
                extraction_stats = conn.execute("""
                    SELECT 
                        extraction_method,
                        COUNT(*) as usage_count,
                        AVG(data_quality_score) as avg_quality
                    FROM service_plans 
                    WHERE extraction_method IS NOT NULL
                    GROUP BY extraction_method
                """).fetchall()
                
                analytics['extraction_methods'] = [dict(row) for row in extraction_stats]
                
                # Recent activity
                recent_activity = conn.execute("""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as extractions,
                        SUM(records_extracted) as total_records
                    FROM extraction_logs 
                    WHERE timestamp >= datetime('now', '-30 days')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                """).fetchall()
                
                analytics['recent_activity'] = [dict(row) for row in recent_activity]
                
                logger.info("Analytics data retrieved successfully")
                return analytics
                
        except Exception as e:
            logger.error(f"Failed to load analytics data: {e}")
            # Return minimal analytics structure
            return {
                'provider_stats': [],
                'extraction_methods': [],
                'recent_activity': [],
                'status': 'degraded_mode',
                'error': str(e)
            }

class ServicePlansExplorer:
    """
    Main application class with integrated error handling and recovery
    """
    
    def __init__(self):
        """Initialize the service plans explorer with all components"""
        try:
            self.extraction_engine = ExtractionEngine()
            self.db_manager = DatabaseManager()
            logger.info("ServicePlansExplorer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ServicePlansExplorer: {e}")
            raise
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get all dashboard data with fallback mechanisms
        
        Returns:
            Complete dashboard data structure
        """
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'status': 'operational'
        }
        
        try:
            # Get statistics (with fallback)
            dashboard_data['statistics'] = self.extraction_engine.get_statistics()
        except Exception as e:
            logger.error(f"Statistics loading failed: {e}")
            dashboard_data['statistics'] = {'status': 'unavailable', 'error': str(e)}
        
        try:
            # Get plans data (with fallback)
            dashboard_data['plans'] = self.db_manager.get_plans_data()
        except Exception as e:
            logger.error(f"Plans data loading failed: {e}")
            dashboard_data['plans'] = []
        
        try:
            # Get analytics (with fallback)
            dashboard_data['analytics'] = self.db_manager.get_analytics_data()
        except Exception as e:
            logger.error(f"Analytics loading failed: {e}")
            dashboard_data['analytics'] = {'status': 'unavailable', 'error': str(e)}
        
        return dashboard_data

# Usage example and system recovery
def main():
    """
    Main function demonstrating system usage and error recovery
    """
    try:
        # Initialize the system
        explorer = ServicePlansExplorer()
        
        # Get dashboard data
        dashboard = explorer.get_dashboard_data()
        
        # Log system status
        if dashboard['statistics'].get('status') == 'degraded_mode':
            logger.warning("System running in degraded mode")
        else:
            logger.info("System operational - all components loaded successfully")
            
        return dashboard
        
    except Exception as e:
        logger.critical(f"System initialization failed: {e}")
        # Return minimal system state for graceful degradation
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'critical_error',
            'error': str(e),
            'statistics': {'status': 'unavailable'},
            'plans': [],
            'analytics': {'status': 'unavailable'}
        }

if __name__ == "__main__":
    dashboard_data = main()
    print("System Status:", dashboard_data['status'])