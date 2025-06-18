"""
Comprehensive test suite for Norwegian Service Aggregator
Unit and integration tests with mocking and fixtures
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from pathlib import Path

# Import modules under test
from core.extractor_engine import (
    ExtractionEngine, ServicePlan, ExtractionTask, ExtractionStatus,
    WindowsAsyncManager, BrowserManager, DatabaseManager
)


class TestServicePlan:
    """Test cases for ServicePlan data structure"""
    
    def test_service_plan_creation(self):
        """Test ServicePlan object creation with all fields"""
        plan = ServicePlan(
            id="test_plan_123",
            provider="Telia",
            name="Smart Mobile Plan",
            category="mobile",
            monthly_price=299.0,
            features=["Unlimited calls", "10GB data", "EU roaming"],
            url="https://telia.no/plan",
            extracted_at="2025-06-18T04:34:49",
            confidence=0.95,
            extraction_method="playwright",
            data_source="web",
            validation_score=0.88
        )
        
        assert plan.id == "test_plan_123"
        assert plan.provider == "Telia"
        assert plan.name == "Smart Mobile Plan"
        assert plan.category == "mobile"
        assert plan.monthly_price == 299.0
        assert len(plan.features) == 3
        assert plan.confidence == 0.95
        assert plan.extraction_method == "playwright"
    
    def test_service_plan_defaults(self):
        """Test ServicePlan with minimal required fields"""
        plan = ServicePlan(
            id="minimal_plan",
            provider="Test Provider",
            name="Test Plan",
            category="test",
            monthly_price=100.0
        )
        
        assert plan.features == []
        assert plan.url == ""
        assert plan.extracted_at == ""
        assert plan.confidence == 1.0
        assert plan.extraction_method == "unknown"
        assert plan.data_source == "web"
        assert plan.validation_score == 1.0


class TestExtractionTask:
    """Test cases for ExtractionTask data structure"""
    
    def test_extraction_task_creation(self):
        """Test ExtractionTask object creation"""
        task = ExtractionTask(
            task_id="task_123",
            url="https://example.no/pricing",
            provider_name="Example Provider"
        )
        
        assert task.task_id == "task_123"
        assert task.url == "https://example.no/pricing"
        assert task.provider_name == "Example Provider"
        assert task.status == ExtractionStatus.PENDING
        assert task.progress == 0.0
        assert task.message == "Task created"
        assert task.result == []
        assert task.error is None
        assert task.retry_count == 0
        assert task.max_retries == 3
    
    def test_extraction_task_status_update(self):
        """Test ExtractionTask status updates"""
        task = ExtractionTask(
            task_id="status_test",
            url="https://test.no",
            provider_name="Test"
        )
        
        # Update to running
        task.status = ExtractionStatus.RUNNING
        task.progress = 0.5
        task.message = "Extracting data"
        task.started_at = datetime.now()
        
        assert task.status == ExtractionStatus.RUNNING
        assert task.progress == 0.5
        assert task.message == "Extracting data"
        assert task.started_at is not None
        
        # Update to completed
        task.status = ExtractionStatus.COMPLETED
        task.progress = 1.0
        task.completed_at = datetime.now()
        task.result = [Mock(spec=ServicePlan)]
        
        assert task.status == ExtractionStatus.COMPLETED
        assert task.progress == 1.0
        assert len(task.result) == 1


class TestWindowsAsyncManager:
    """Test cases for Windows async compatibility manager"""
    
    @pytest.fixture
    def async_manager(self):
        """Create WindowsAsyncManager instance for testing"""
        return WindowsAsyncManager()
    
    def test_async_manager_initialization(self, async_manager):
        """Test proper initialization of async manager"""
        assert async_manager._loop is None
        assert async_manager._loop_thread is None
        assert not async_manager._shutdown_event.is_set()
    
    @pytest.mark.asyncio
    async def test_simple_coroutine_execution(self, async_manager):
        """Test execution of simple coroutine"""
        async def simple_task():
            await asyncio.sleep(0.1)
            return "success"
        
        # Start the background loop
        async_manager.start_background_loop()
        
        # Execute coroutine
        result = async_manager.run_coroutine(simple_task())
        assert result == "success"
        
        # Cleanup
        async_manager.shutdown()
    
    def test_async_manager_shutdown(self, async_manager):
        """Test proper shutdown of async manager"""
        async_manager.start_background_loop()
        assert async_manager._loop is not None
        
        async_manager.shutdown()
        # Allow time for shutdown
        time.sleep(1)
        assert async_manager._shutdown_event.is_set()


class TestBrowserManager:
    """Test cases for browser management"""
    
    @pytest.fixture
    def browser_manager(self):
        """Create BrowserManager instance for testing"""
        return BrowserManager(max_browsers=2)
    
    def test_browser_manager_initialization(self, browser_manager):
        """Test proper initialization of browser manager"""
        assert browser_manager._max_browsers == 2
        assert len(browser_manager._active_browsers) == 0
    
    def test_user_agent_generation(self, browser_manager):
        """Test user agent generation"""
        user_agent = browser_manager._get_random_user_agent()
        assert isinstance(user_agent, str)
        assert "Mozilla" in user_agent
        assert "Chrome" in user_agent or "Firefox" in user_agent
    
    def test_stealth_headers_generation(self, browser_manager):
        """Test stealth headers generation"""
        headers = browser_manager._get_stealth_headers()
        assert isinstance(headers, dict)
        assert "Accept" in headers
        assert "Accept-Language" in headers
        assert "no" in headers["Accept-Language"]  # Norwegian language preference
    
    def test_active_browser_count(self, browser_manager):
        """Test active browser count tracking"""
        assert browser_manager.get_active_browser_count() == 0
        
        # Simulate adding browsers
        mock_browser = Mock()
        browser_manager._active_browsers.add(mock_browser)
        assert browser_manager.get_active_browser_count() == 1


class TestDatabaseManager:
    """Test cases for database management"""
    
    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create temporary database path for testing"""
        return str(tmp_path / "test_services.db")
    
    @pytest.fixture
    def db_manager(self, temp_db_path):
        """Create DatabaseManager instance with temporary database"""
        return DatabaseManager(db_path=temp_db_path)
    
    def test_database_initialization(self, db_manager):
        """Test database table creation and indexes"""
        import sqlite3
        
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        # Check if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert "plans" in tables
        assert "extraction_tasks" in tables
        assert "extraction_logs" in tables
        
        # Check if indexes exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]
        
        assert any("idx_provider" in idx for idx in indexes)
        assert any("idx_category" in idx for idx in indexes)
        
        conn.close()
    
    def test_store_plans(self, db_manager):
        """Test storing service plans in database"""
        test_plans = [
            ServicePlan(
                id="test_plan_1",
                provider="Telia",
                name="Test Plan 1",
                category="mobile",
                monthly_price=299.0,
                features=["Feature 1", "Feature 2"],
                url="https://telia.no/plan1",
                extracted_at=datetime.now().isoformat(),
                confidence=0.9
            ),
            ServicePlan(
                id="test_plan_2",
                provider="Telenor",
                name="Test Plan 2",
                category="mobile",
                monthly_price=399.0,
                features=["Feature A", "Feature B", "Feature C"],
                url="https://telenor.no/plan2",
                extracted_at=datetime.now().isoformat(),
                confidence=0.85
            )
        ]
        
        # Store plans
        db_manager.store_plans(test_plans)
        
        # Verify storage
        stored_plans = db_manager.get_all_plans()
        assert len(stored_plans) == 2
        
        # Check first plan
        plan1 = next(p for p in stored_plans if p['id'] == 'test_plan_1')
        assert plan1['provider'] == 'Telia'
        assert plan1['name'] == 'Test Plan 1'
        assert plan1['monthly_price'] == 299.0
        assert len(plan1['features']) == 2
    
    def test_get_statistics(self, db_manager):
        """Test statistics generation"""
        # Add test data
        test_plans = [
            ServicePlan(
                id="stats_plan_1",
                provider="Telia",
                name="Stats Plan 1",
                category="mobile",
                monthly_price=299.0,
                features=["Feature"],
                url="https://telia.no",
                extracted_at=datetime.now().isoformat(),
                confidence=0.9
            ),
            ServicePlan(
                id="stats_plan_2",
                provider="Telenor",
                name="Stats Plan 2",
                category="electricity",
                monthly_price=150.0,
                features=["Feature"],
                url="https://telenor.no",
                extracted_at=datetime.now().isoformat(),
                confidence=0.8
            )
        ]
        
        db_manager.store_plans(test_plans)
        
        # Get statistics
        stats = db_manager.get_statistics()
        
        assert stats['total_plans'] == 2
        assert len(stats['providers']) == 2
        assert len(stats['categories']) == 2
        assert 'Telia' in stats['providers']
        assert 'Telenor' in stats['providers']
        assert 'mobile' in stats['categories']
        assert 'electricity' in stats['categories']
    
    def test_plan_validation(self, db_manager):
        """Test plan validation before storage"""
        # Valid plan
        valid_plan = ServicePlan(
            id="valid_plan",
            provider="TestProvider",
            name="Valid Plan",
            category="test",
            monthly_price=100.0
        )
        assert db_manager._validate_plan(valid_plan) == True
        
        # Invalid plan - missing required fields
        invalid_plan = ServicePlan(
            id="",  # Empty ID
            provider="TestProvider",
            name="",  # Empty name
            category="test",
            monthly_price=100.0
        )
        assert db_manager._validate_plan(invalid_plan) == False
        
        # Invalid plan - negative price
        negative_price_plan = ServicePlan(
            id="negative_plan",
            provider="TestProvider",
            name="Negative Plan",
            category="test",
            monthly_price=-50.0  # Negative price
        )
        assert db_manager._validate_plan(negative_price_plan) == False


class TestExtractionEngine:
    """Integration tests for the main extraction engine"""
    
    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create temporary database path for testing"""
        return str(tmp_path / "test_engine.db")
    
    @pytest.fixture
    def extraction_engine(self, temp_db_path):
        """Create ExtractionEngine instance for testing"""
        with patch('core.extractor_engine.DatabaseManager') as mock_db_manager:
            mock_db_manager.return_value.db_path = temp_db_path
            engine = ExtractionEngine()
            return engine
    
    def test_extraction_patterns_loading(self, extraction_engine):
        """Test loading of extraction patterns"""
        patterns = extraction_engine.patterns
        
        assert isinstance(patterns, dict)
        assert "telia.no" in patterns
        assert "telenor.no" in patterns
        assert "ice.no" in patterns
        assert "talkmore.no" in patterns
        assert "fortum.no" in patterns
        
        # Check pattern structure
        telia_pattern = patterns["telia.no"]
        assert "category" in telia_pattern
        assert "price_regex" in telia_pattern
        assert "name_selectors" in telia_pattern
        assert "container_selectors" in telia_pattern
        assert "features_selectors" in telia_pattern
        
        assert telia_pattern["category"] == "mobile"
    
    def test_domain_extraction(self, extraction_engine):
        """Test URL domain extraction"""
        test_cases = [
            ("https://www.telia.no/privat/mobil", "www.telia.no"),
            ("http://telenor.no/pricing", "telenor.no"),
            ("https://example.com:8080/path", "example.com:8080"),
            ("www.test.no", "www.test.no")
        ]
        
        for url, expected_domain in test_cases:
            domain = extraction_engine._extract_domain(url)
            assert domain == expected_domain
    
    def test_task_submission(self, extraction_engine):
        """Test extraction task submission"""
        url = "https://test.no/pricing"
        provider_name = "Test Provider"
        
        with patch.object(extraction_engine.task_queue, 'put') as mock_put:
            task_id = extraction_engine.submit_extraction_task(url, provider_name)
            
            assert isinstance(task_id, str)
            assert len(task_id) > 0
            assert task_id in extraction_engine.active_tasks
            
            task = extraction_engine.active_tasks[task_id]
            assert task.url == url
            assert task.provider_name == provider_name
            assert task.status == ExtractionStatus.PENDING
            
            # Verify task was queued
            mock_put.assert_called_once()
    
    def test_invalid_url_submission(self, extraction_engine):
        """Test submission of invalid URLs"""
        invalid_urls = [
            "",  # Empty URL
            "not-a-url",  # Invalid format
            "ftp://example.com",  # Wrong protocol
            None  # None value
        ]
        
        for invalid_url in invalid_urls:
            with pytest.raises(ValueError):
                extraction_engine.submit_extraction_task(invalid_url, "Provider")
    
    def test_health_status(self, extraction_engine):
        """Test system health status reporting"""
        health_status = extraction_engine.get_health_status()
        
        assert isinstance(health_status, dict)
        assert 'status' in health_status
        assert 'timestamp' in health_status
        assert 'components' in health_status
        assert 'metrics' in health_status
        
        # Check component health
        components = health_status['components']
        assert 'database' in components
        assert 'task_queue' in components
        assert 'browsers' in components
        assert 'workers' in components
        
        # Check metrics
        metrics = health_status['metrics']
        assert 'active_tasks' in metrics
        assert 'queue_size' in metrics
    
    @patch('core.extractor_engine.requests.get')
    def test_fallback_extraction(self, mock_get, extraction_engine):
        """Test fallback extraction with mocked HTTP response"""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <body>
                <div class="plan-card">
                    <h3>Test Mobile Plan</h3>
                    <div class="price">299 kr</div>
                    <ul>
                        <li>Unlimited calls</li>
                        <li>10GB data</li>
                    </ul>
                </div>
            </body>
        </html>
        """
        mock_response.headers = {'content-type': 'text/html'}
        mock_get.return_value = mock_response
        
        # Test fallback extraction
        url = "https://test.no/pricing"
        domain = "test.no"
        
        plans = extraction_engine._enhanced_fallback_extraction(url, domain)
        
        # Verify HTTP request was made
        mock_get.assert_called_once()
        
        # Check if plans were extracted (depends on generic extraction logic)
        assert isinstance(plans, list)
    
    def test_plan_signature_creation(self, extraction_engine):
        """Test plan signature creation for deduplication"""
        plan1 = ServicePlan(
            id="plan1",
            provider="Telia",
            name="Smart Plan",
            category="mobile",
            monthly_price=299.0
        )
        
        plan2 = ServicePlan(
            id="plan2",
            provider="Telia", 
            name="Smart Plan",  # Same name
            category="mobile",
            monthly_price=299.0  # Same price
        )
        
        plan3 = ServicePlan(
            id="plan3",
            provider="Telia",
            name="Premium Plan",  # Different name
            category="mobile",
            monthly_price=299.0
        )
        
        sig1 = extraction_engine._create_plan_signature(plan1)
        sig2 = extraction_engine._create_plan_signature(plan2)
        sig3 = extraction_engine._create_plan_signature(plan3)
        
        # Plans with same name and price should have same signature
        assert sig1 == sig2
        
        # Plan with different name should have different signature
        assert sig1 != sig3
    
    def test_plan_quality_validation(self, extraction_engine):
        """Test plan quality validation"""
        # High quality plan
        high_quality_plan = ServicePlan(
            id="high_quality",
            provider="Telia",
            name="Excellent Mobile Plan",
            category="mobile",
            monthly_price=299.0,
            features=["Feature 1", "Feature 2"],
            confidence=0.95
        )
        assert extraction_engine._validate_plan_quality(high_quality_plan) == True
        
        # Low quality plan - low confidence
        low_confidence_plan = ServicePlan(
            id="low_confidence",
            provider="Telia",
            name="Uncertain Plan",
            category="mobile",
            monthly_price=299.0,
            confidence=0.2  # Very low confidence
        )
        assert extraction_engine._validate_plan_quality(low_confidence_plan) == False
        
        # Invalid plan - generic name
        generic_plan = ServicePlan(
            id="generic",
            provider="Telia",
            name="plan",  # Too generic
            category="mobile",
            monthly_price=299.0,
            confidence=0.9
        )
        assert extraction_engine._validate_plan_quality(generic_plan) == False
    
    def test_shutdown(self, extraction_engine):
        """Test graceful shutdown of extraction engine"""
        # Start the engine components
        extraction_engine.async_manager.start_background_loop()
        
        # Shutdown
        extraction_engine.shutdown()
        
        # Verify shutdown state
        assert extraction_engine._shutdown_event.is_set()
        assert extraction_engine.async_manager._shutdown_event.is_set()


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        "-v",
        "--cov=core",
        "--cov-report=html",
        "--cov-report=term-missing",
        __file__
    ])