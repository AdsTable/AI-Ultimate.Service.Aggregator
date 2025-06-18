# core/extractor_engine.py
"""
Production-grade web extraction engine with Windows compatibility
Implements background task processing and robust error handling
"""

import asyncio
import platform
import threading
import sqlite3
import json
import time
import random
import hashlib
import requests
import re
import queue
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum
import uuid


class ExtractionStatus(Enum):
    """Enumeration for extraction task status tracking"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ServicePlan:
    """Standardized service plan data structure"""
    id: str
    provider: str
    name: str
    category: str
    monthly_price: float
    features: List[str] = field(default_factory=list)
    url: str = ""
    extracted_at: str = ""
    confidence: float = 1.0
    extraction_method: str = "unknown"


@dataclass
class ExtractionTask:
    """Background extraction task with status tracking"""
    task_id: str
    url: str
    provider_name: str
    status: ExtractionStatus = ExtractionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: str = "Task created"
    result: List[ServicePlan] = field(default_factory=list)
    error: Optional[str] = None


class AsyncEventLoopManager:
    """Manages isolated async event loops for Windows compatibility"""
    
    def __init__(self):
        self._setup_windows_compatibility()
        self._loop_thread = None
        self._loop = None
        self._shutdown_event = threading.Event()
    
    def _setup_windows_compatibility(self) -> None:
        """Configure asyncio policy for Windows subprocess support"""
        if platform.system() == 'Windows':
            # Use WindowsSelectorEventLoopPolicy for subprocess support
            if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
            # Set specific event loop for this thread
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            except RuntimeError:
                pass  # Loop already exists
    
    def start_background_loop(self) -> None:
        """Start dedicated background event loop in separate thread"""
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        
        self._loop_thread = threading.Thread(
            target=self._run_background_loop,
            name="AsyncEventLoop",
            daemon=True
        )
        self._loop_thread.start()
        
        # Wait for loop to be ready
        timeout = 10.0
        start_time = time.time()
        while self._loop is None and (time.time() - start_time) < timeout:
            time.sleep(0.1)
    
    def _run_background_loop(self) -> None:
        """Run the background event loop"""
        try:
            # Create new event loop for this thread
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            # Run until shutdown
            self._loop.run_until_complete(self._loop_runner())
        except Exception as e:
            logging.error(f"Background loop error: {e}")
        finally:
            if self._loop:
                self._loop.close()
    
    async def _loop_runner(self) -> None:
        """Main loop runner that waits for shutdown signal"""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(0.1)
    
    def run_coroutine(self, coro) -> Any:
        """Execute coroutine in background loop thread-safely"""
        if self._loop is None:
            self.start_background_loop()
        
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=120)  # 2 minute timeout
    
    def shutdown(self) -> None:
        """Shutdown the background event loop"""
        self._shutdown_event.set()
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)


class BrowserManager:
    """Manages browser instances with proper resource cleanup"""
    
    def __init__(self):
        self._browser_pool = queue.Queue(maxsize=3)  # Limit concurrent browsers
        self._active_browsers = set()
        self._lock = threading.Lock()
    
    async def get_browser_context(self):
        """Get browser context with stealth configuration"""
        try:
            from playwright.async_api import async_playwright
            
            playwright = await async_playwright().__aenter__()
            
            # Enhanced browser launch for Windows
            browser = await playwright.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-extensions',
                    '--disable-gpu',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--disable-features=TranslateUI',
                    '--disable-ipc-flooding-protection',
                    '--single-process',  # Critical for Windows
                    '--no-zygote',       # Additional Windows compatibility
                    '--disable-dev-tools',
                    '--disable-background-networking'
                ]
            )
            
            # Create stealth context
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent=self._get_random_user_agent(),
                extra_http_headers=self._get_stealth_headers(),
                ignore_https_errors=True,
                java_script_enabled=True
            )
            
            with self._lock:
                self._active_browsers.add(browser)
            
            return playwright, browser, context
            
        except Exception as e:
            logging.error(f"Browser initialization failed: {e}")
            raise
    
    async def cleanup_browser(self, playwright, browser, context):
        """Properly cleanup browser resources"""
        try:
            if context:
                await context.close()
            if browser:
                await browser.close()
                with self._lock:
                    self._active_browsers.discard(browser)
            if playwright:
                await playwright.__aexit__(None, None, None)
        except Exception as e:
            logging.error(f"Browser cleanup error: {e}")
    
    def _get_random_user_agent(self) -> str:
        """Generate realistic user agent for stealth"""
        agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0'
        ]
        return random.choice(agents)
    
    def _get_stealth_headers(self) -> Dict[str, str]:
        """Generate stealth HTTP headers"""
        return {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,no;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }


class DatabaseManager:
    """Thread-safe database operations with connection pooling"""
    
    def __init__(self, db_path: str = "services.db"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize database with optimized schema"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS plans (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                monthly_price REAL DEFAULT 0.0,
                features TEXT DEFAULT '[]',
                url TEXT DEFAULT '',
                extracted_at TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                extraction_method TEXT DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS extraction_tasks (
                task_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                progress REAL DEFAULT 0.0,
                message TEXT DEFAULT '',
                error TEXT,
                result_count INTEGER DEFAULT 0
            )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_provider ON plans(provider)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON plans(category)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_price ON plans(monthly_price)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_extracted ON plans(extracted_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_status ON extraction_tasks(status)")
            
            conn.commit()
            conn.close()
            logging.info("Database initialized successfully")
    
    def store_plans(self, plans: List[ServicePlan]) -> None:
        """Store service plans with thread safety"""
        if not plans:
            return
        
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                for plan in plans:
                    conn.execute("""
                    INSERT OR REPLACE INTO plans 
                    (id, provider, name, category, monthly_price, features, url, 
                     extracted_at, confidence, extraction_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        plan.id, plan.provider, plan.name, plan.category,
                        plan.monthly_price, json.dumps(plan.features, ensure_ascii=False),
                        plan.url, plan.extracted_at, plan.confidence, plan.extraction_method
                    ))
                
                conn.commit()
                logging.info(f"Stored {len(plans)} plans in database")
            finally:
                conn.close()
    
    def update_task(self, task: ExtractionTask) -> None:
        """Update extraction task status"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                INSERT OR REPLACE INTO extraction_tasks 
                (task_id, url, provider_name, status, created_at, started_at, 
                 completed_at, progress, message, error, result_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.task_id, task.url, task.provider_name, task.status.value,
                    task.created_at.isoformat(), 
                    task.started_at.isoformat() if task.started_at else None,
                    task.completed_at.isoformat() if task.completed_at else None,
                    task.progress, task.message, task.error, len(task.result)
                ))
                conn.commit()
            finally:
                conn.close()
    
    def get_all_plans(self) -> List[Dict[str, Any]]:
        """Retrieve all plans with thread safety"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("""
                SELECT id, provider, name, category, monthly_price, features, 
                       url, extracted_at, confidence, extraction_method
                FROM plans ORDER BY extracted_at DESC
                """)
                
                plans = []
                for row in cursor.fetchall():
                    plans.append({
                        'id': row[0],
                        'provider': row[1],
                        'name': row[2],
                        'category': row[3],
                        'monthly_price': row[4],
                        'features': json.loads(row[5]) if row[5] else [],
                        'url': row[6],
                        'extracted_at': row[7],
                        'confidence': row[8],
                        'extraction_method': row[9]
                    })
                
                return plans
            finally:
                conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                # Total plans
                total_plans = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
                
                # Plans by category
                category_stats = conn.execute("""
                    SELECT category, COUNT(*) as count, 
                           ROUND(AVG(monthly_price), 2) as avg_price
                    FROM plans WHERE monthly_price > 0
                    GROUP BY category
                """).fetchall()
                
                # Plans by provider
                provider_stats = conn.execute("""
                    SELECT provider, COUNT(*) as count
                    FROM plans GROUP BY provider ORDER BY count DESC
                """).fetchall()
                
                # Recent extractions
                recent_extractions = conn.execute("""
                    SELECT COUNT(*) FROM plans
                    WHERE extracted_at > datetime('now', '-24 hours')
                """).fetchone()[0]
                
                # Task statistics
                task_stats = conn.execute("""
                    SELECT status, COUNT(*) as count
                    FROM extraction_tasks
                    GROUP BY status
                """).fetchall()
                
                return {
                    "total_plans": total_plans,
                    "categories": {row[0]: {"count": row[1], "avg_price": row[2]} 
                                 for row in category_stats},
                    "providers": {row[0]: row[1] for row in provider_stats},
                    "recent_extractions": recent_extractions,
                    "task_stats": {row[0]: row[1] for row in task_stats}
                }
            finally:
                conn.close()


class ExtractionEngine:
    """Main extraction engine with background task processing"""
    
    def __init__(self):
        self.loop_manager = AsyncEventLoopManager()
        self.browser_manager = BrowserManager()
        self.db_manager = DatabaseManager()
        self.task_queue = queue.Queue()
        self.active_tasks: Dict[str, ExtractionTask] = {}
        self.worker_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ExtractWorker")
        self._shutdown_event = threading.Event()
        
        # Load extraction patterns
        self.patterns = self._load_extraction_patterns()
        
        # Start background workers
        self._start_workers()
        
        logging.info("✅ Extraction Engine initialized")
    
    def _load_extraction_patterns(self) -> Dict[str, Any]:
        """Load proven extraction patterns for Norwegian providers"""
        return {
            "telia.no": {
                "category": "mobile",
                "price_regex": r'(\d+)\s*kr',
                "name_selectors": ['h3', '.plan-name', '.subscription-title', '[data-testid*="plan"]'],
                "container_selectors": ['.plan-card', '.subscription-box', '.product-card', '.price-card'],
                "features_selectors": ['li', '.feature', '.benefit', '.included'],
                "wait_selectors": ['.plan-card', '.subscription-box']
            },
            "telenor.no": {
                "category": "mobile",
                "price_regex": r'(\d+)\s*kr',
                "name_selectors": ['h3', '.plan-title', '[data-cy*="plan"]'],
                "container_selectors": ['.product-card', '.plan-container', '.mobile-plan'],
                "features_selectors": ['li', '.feature-list li', '.benefits li'],
                "wait_selectors": ['.product-card', '.plan-container']
            },
            "ice.no": {
                "category": "mobile",
                "price_regex": r'(\d+)\s*kr',
                "name_selectors": ['h3', '.plan-name', '.title'],
                "container_selectors": ['.plan-box', '.offer-card'],
                "features_selectors": ['li', '.features li'],
                "wait_selectors": ['.plan-box']
            },
            "talkmore.no": {
                "category": "mobile",
                "price_regex": r'(\d+)\s*kr',
                "name_selectors": ['h3', '.plan-title', '.price-title', '.subscription-name'],
                "container_selectors": ['.plan-card', '.price-box', '.subscription', '.offer-item'],
                "features_selectors": ['li', '.feature', '.benefit'],
                "wait_selectors": ['.plan-card', '.subscription']
            },
            "fortum.no": {
                "category": "electricity",
                "price_regex": r'(\d+[,\.]\d+)\s*øre',
                "name_selectors": ['h3', '.plan-title', '.product-name'],
                "container_selectors": ['.price-plan', '.electricity-plan'],
                "features_selectors": ['li', '.feature'],
                "wait_selectors": ['.price-plan']
            }
        }
    
    def _start_workers(self) -> None:
        """Start background worker threads"""
        self.loop_manager.start_background_loop()
        
        # Start task processing workers
        for i in range(2):
            self.worker_pool.submit(self._worker_thread, f"worker-{i}")
    
    def _worker_thread(self, worker_name: str) -> None:
        """Background worker thread for processing extraction tasks"""
        logging.info(f"Worker {worker_name} started")
        
        while not self._shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                task = self.task_queue.get(timeout=1.0)
                
                if task is None:  # Shutdown signal
                    break
                
                # Process the task
                self._process_extraction_task(task, worker_name)
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Worker {worker_name} error: {e}")
    
    def _process_extraction_task(self, task: ExtractionTask, worker_name: str) -> None:
        """Process a single extraction task"""
        logging.info(f"Worker {worker_name} processing task {task.task_id}")
        
        try:
            # Update task status
            task.status = ExtractionStatus.RUNNING
            task.started_at = datetime.now()
            task.progress = 0.1
            task.message = f"Starting extraction from {task.provider_name}"
            self.db_manager.update_task(task)
            
            # Execute extraction using the async event loop
            result = self.loop_manager.run_coroutine(
                self._extract_from_provider_async(task)
            )
            
            # Update task with results
            task.result = result
            task.status = ExtractionStatus.COMPLETED
            task.completed_at = datetime.now()
            task.progress = 1.0
            task.message = f"Completed: {len(result)} plans extracted"
            
            # Store results in database
            if result:
                self.db_manager.store_plans(result)
            
        except Exception as e:
            # Handle task failure
            task.status = ExtractionStatus.FAILED
            task.completed_at = datetime.now()
            task.progress = 0.0
            task.error = str(e)
            task.message = f"Extraction failed: {str(e)}"
            logging.error(f"Task {task.task_id} failed: {e}")
        
        finally:
            # Update task in database
            self.db_manager.update_task(task)
            
            # Remove from active tasks
            self.active_tasks.pop(task.task_id, None)
    
    async def _extract_from_provider_async(self, task: ExtractionTask) -> List[ServicePlan]:
        """Async extraction method with comprehensive error handling"""
        domain = self._extract_domain(task.url)
        
        # Update progress
        task.progress = 0.2
        task.message = f"Initializing browser for {domain}"
        self.db_manager.update_task(task)
        
        try:
            # Try Playwright extraction
            plans = await self._playwright_extraction(task, domain)
            
            if plans:
                task.progress = 0.9
                task.message = f"Successfully extracted {len(plans)} plans"
                return plans
            else:
                # Fallback to requests-based extraction
                task.progress = 0.5
                task.message = "Trying fallback extraction method"
                self.db_manager.update_task(task)
                
                return self._fallback_extraction(task.url, domain)
                
        except Exception as e:
            logging.error(f"Async extraction failed for {domain}: {e}")
            # Try fallback as last resort
            return self._fallback_extraction(task.url, domain)
    
    async def _playwright_extraction(self, task: ExtractionTask, domain: str) -> List[ServicePlan]:
        """Enhanced Playwright-based extraction with comprehensive stealth and error handling"""
        playwright = None
        browser = None
        context = None
        
        try:
            # Get browser context with resource management
            playwright, browser, context = await self.browser_manager.get_browser_context()
            page = await context.new_page()
            
            # Update progress
            task.progress = 0.35
            task.message = f"Browser ready, navigating to {domain}"
            self.db_manager.update_task(task)
            
            # Apply comprehensive stealth measures
            await self._apply_comprehensive_stealth(page, domain)
            
            # Navigate and extract with enhanced error handling
            plans = await self._navigate_and_extract_comprehensive(page, task, domain)
            
            return plans
            
        except Exception as e:
            logger.error(f"Playwright extraction failed for {domain}: {e}", exc_info=True)
            return []
        
        finally:
            # Ensure proper cleanup
            try:
                if 'page' in locals():
                    await page.close()
                await self.browser_manager.cleanup_browser(playwright, browser, context)
            except Exception as cleanup_error:
                logger.error(f"Browser cleanup error: {cleanup_error}")
    
    async def _apply_comprehensive_stealth(self, page, domain: str) -> None:
        """Apply comprehensive stealth measures with domain-specific optimizations"""
        
        # Core stealth script with enhanced anti-detection
        stealth_script = """
        // Remove automation signatures
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined,
        });
        
        // Mock realistic plugins array
        Object.defineProperty(navigator, 'plugins', {
            get: () => [
                {name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format'},
                {name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: 'PDF Viewer'},
                {name: 'Native Client', filename: 'internal-nacl-plugin', description: 'Native Client'}
            ],
        });
        
        // Mock realistic languages
        Object.defineProperty(navigator, 'languages', {
            get: () => ['no', 'nb', 'en-US', 'en'],
        });
        
        // Override permissions API
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
        );
        
        // Mock realistic screen properties
        Object.defineProperty(screen, 'colorDepth', {
            get: () => 24,
        });
        
        Object.defineProperty(screen, 'pixelDepth', {
            get: () => 24,
        });
        
        // Mock realistic connection
        Object.defineProperty(navigator, 'connection', {
            get: () => ({
                effectiveType: '4g',
                rtt: 100,
                downlink: 10
            }),
        });
        
        // Override chrome runtime
        if (window.chrome) {
            Object.defineProperty(window.chrome, 'runtime', {
                get: () => ({
                    onConnect: undefined,
                    onMessage: undefined
                }),
            });
        }
        """
        
        await page.add_init_script(stealth_script)
        
        # Block tracking, analytics, and ads for better performance and stealth
        blocked_domains = [
            'google-analytics.com', 'googletagmanager.com', 'facebook.com/tr',
            'hotjar.com', 'fullstory.com', 'recaptcha.net', 'doubleclick.net',
            'googlesyndication.com', 'amazon-adsystem.com', 'google.com/pagead',
            'googleadservices.com', 'bing.com', 'microsoft.com/en-us/bing',
            'youtube.com', 'twitter.com', 'linkedin.com', 'instagram.com'
        ]
        
        await page.route("**/*", lambda route: (
            route.abort() if any(domain in route.request.url for domain in blocked_domains)
            else route.continue_()
        ))
        
        # Set additional headers for Norwegian context
        await page.set_extra_http_headers({
            'Accept-Language': 'no,nb;q=0.9,en-US;q=0.8,en;q=0.7',
            'sec-ch-ua-platform': '"Windows"',
            'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120"'
        })
    
    async def _navigate_and_extract_comprehensive(self, page, task: ExtractionTask, domain: str) -> List[ServicePlan]:
        """Enhanced navigation and extraction with intelligent waiting and error recovery"""
        
        try:
            # Phase 1: Initial navigation
            task.progress = 0.4
            task.message = f"Navigating to {domain}"
            self.db_manager.update_task(task)
            
            # Navigate with multiple wait strategies
            response = await page.goto(
                task.url,
                wait_until='domcontentloaded',
                timeout=60000  # Increased timeout for slow sites
            )
            
            # Check response status
            if response and response.status >= 400:
                logger.warning(f"HTTP {response.status} received for {task.url}")
                if response.status >= 500:
                    raise Exception(f"Server error: HTTP {response.status}")
            
            # Phase 2: Handle overlays and consent
            task.progress = 0.45
            task.message = "Handling page overlays and consent dialogs"
            self.db_manager.update_task(task)
            
            await self._handle_comprehensive_overlays(page, domain)
            
            # Phase 3: Smart content waiting
            task.progress = 0.5
            task.message = "Waiting for content to load"
            self.db_manager.update_task(task)
            
            await self._smart_content_waiting(page, domain)
            
            # Phase 4: Human behavior simulation
            task.progress = 0.6
            task.message = "Simulating human browsing behavior"
            self.db_manager.update_task(task)
            
            await self._simulate_advanced_behavior(page)
            
            # Phase 5: Content extraction
            task.progress = 0.7
            task.message = f"Extracting service plans from {domain}"
            self.db_manager.update_task(task)
            
            # Get page content and extract
            html_content = await page.content()
            plans = self._extract_plans_from_html_enhanced(html_content, domain, task.url)
            
            # Phase 6: Result validation
            task.progress = 0.9
            task.message = f"Validating {len(plans)} extracted plans"
            self.db_manager.update_task(task)
            
            validated_plans = self._validate_and_enhance_plans(plans, domain)
            
            return validated_plans
            
        except Exception as e:
            logger.error(f"Comprehensive extraction failed for {domain}: {e}")
            raise
    
    async def _handle_comprehensive_overlays(self, page, domain: str) -> None:
        """Handle various types of overlays with Norwegian-specific patterns"""
        
        # Norwegian cookie consent patterns
        norwegian_patterns = [
            'button:has-text("Godta alle")',
            'button:has-text("Godta")',
            'button:has-text("Aksepter alle")',
            'button:has-text("Aksepter")',
            'button:has-text("Jeg forstår")',
            'button:has-text("OK")',
            'button:has-text("Lukk")',
            'button:has-text("Fortsett")'
        ]
        
        # English patterns
        english_patterns = [
            'button:has-text("Accept all")',
            'button:has-text("Accept")',
            'button:has-text("I understand")',
            'button:has-text("Continue")',
            'button:has-text("Close")',
            'button:has-text("OK")'
        ]
        
        # CSS selectors
        css_patterns = [
            '.cookie-accept', '.accept-all-cookies', '.consent-accept',
            '[data-testid*="accept"]', '[data-cy*="accept"]',
            '[id*="CookieConsent"] button', '.cookie-banner button',
            '#cookie-consent button', '.consent-manager button',
            '.gdpr-accept', '.privacy-accept', '.modal-accept'
        ]
        
        all_patterns = norwegian_patterns + english_patterns + css_patterns
        
        # Try each pattern with timeout
        for pattern in all_patterns:
            try:
                await page.click(pattern, timeout=3000)
                await asyncio.sleep(1)
                logger.info(f"Handled overlay with pattern: {pattern}")
                break
            except:
                continue
        
        # Handle age verification if present
        age_patterns = [
            'button:has-text("Ja")', 'button:has-text("Yes")',
            'button:has-text("Jeg er over 18")',
            'select[name*="age"] option[value="1"]'
        ]
        
        for pattern in age_patterns:
            try:
                await page.click(pattern, timeout=2000)
                await asyncio.sleep(0.5)
                logger.info(f"Handled age verification: {pattern}")
                break
            except:
                continue
    
    async def _smart_content_waiting(self, page, domain: str) -> None:
        """Intelligent content waiting based on domain patterns and dynamic detection"""
        
        pattern = self.patterns.get(domain, {})
        wait_selectors = pattern.get('wait_selectors', [])
        
        # Primary strategy: Wait for domain-specific content
        content_loaded = False
        for selector in wait_selectors:
            try:
                await page.wait_for_selector(selector, timeout=15000)
                logger.info(f"Domain-specific content loaded: {selector}")
                content_loaded = True
                break
            except:
                continue
        
        # Secondary strategy: Wait for common plan indicators
        if not content_loaded:
            common_selectors = [
                '[class*="plan"]', '[class*="price"]', '[class*="package"]',
                '[class*="subscription"]', '[data-testid*="plan"]',
                'h3', 'h2', '.card'
            ]
            
            for selector in common_selectors:
                try:
                    await page.wait_for_selector(selector, timeout=10000)
                    logger.info(f"Common content loaded: {selector}")
                    content_loaded = True
                    break
                except:
                    continue
        
        # Wait for network idle as fallback
        if not content_loaded:
            try:
                await page.wait_for_load_state('networkidle', timeout=20000)
                logger.info("Network idle achieved")
            except:
                logger.warning("Network idle timeout, proceeding anyway")
        
        # Additional wait for dynamic content
        await asyncio.sleep(random.uniform(2, 4))
        
        # Check if page is still loading
        try:
            loading_indicators = [
                '.loading', '.spinner', '[class*="load"]',
                '.skeleton', '.placeholder'
            ]
            
            for indicator in loading_indicators:
                elements = await page.query_selector_all(indicator)
                if elements:
                    logger.info(f"Waiting for loading indicator to disappear: {indicator}")
                    await page.wait_for_selector(f'{indicator}:not(:visible)', timeout=10000)
                    break
        except:
            pass
    
    async def _simulate_advanced_behavior(self, page) -> None:
        """Simulate advanced human browsing behavior with realistic patterns"""
        
        try:
            # Get page dimensions
            viewport = await page.evaluate('''() => {
                return {
                    width: window.innerWidth,
                    height: window.innerHeight,
                    scrollHeight: document.body.scrollHeight
                }
            }''')
            
            # Phase 1: Initial mouse movement and hover
            for _ in range(random.randint(2, 4)):
                x = random.randint(100, min(1800, viewport['width'] - 100))
                y = random.randint(100, min(900, viewport['height'] - 100))
                await page.mouse.move(x, y)
                await asyncio.sleep(random.uniform(0.1, 0.3))
            
            # Phase 2: Realistic scrolling with pauses
            scroll_positions = []
            current_scroll = 0
            max_scroll = viewport['scrollHeight'] - viewport['height']
            
            # Generate realistic scroll pattern
            while current_scroll < max_scroll * 0.8:  # Don't scroll to bottom
                scroll_distance = random.randint(200, 600)
                current_scroll += scroll_distance
                scroll_positions.append(min(current_scroll, max_scroll))
            
            for position in scroll_positions:
                await page.evaluate(f'window.scrollTo(0, {position})')
                await asyncio.sleep(random.uniform(1.0, 2.5))
                
                # Occasional pause to "read"
                if random.random() < 0.3:
                    await asyncio.sleep(random.uniform(1.0, 3.0))
            
            # Phase 3: Element interactions
            try:
                # Find interactive elements to hover over
                interactive_elements = await page.query_selector_all(
                    'h1, h2, h3, .plan-card, .price, button:not([type="submit"]), a:not([href^="mailto"])'
                )
                
                if interactive_elements:
                    # Hover over a few elements
                    elements_to_hover = random.sample(
                        interactive_elements, 
                        min(3, len(interactive_elements))
                    )
                    
                    for element in elements_to_hover:
                        try:
                            await element.hover()
                            await asyncio.sleep(random.uniform(0.5, 1.5))
                        except:
                            continue
            except:
                pass
            
            # Phase 4: Random mouse movements to simulate reading
            for _ in range(random.randint(1, 3)):
                x = random.randint(100, min(1800, viewport['width'] - 100))
                y = random.randint(200, min(800, viewport['height'] - 100))
                await page.mouse.move(x, y)
                await asyncio.sleep(random.uniform(0.3, 0.8))
            
        except Exception as e:
            logger.warning(f"Advanced behavior simulation failed: {e}")
    
    def _extract_plans_from_html_enhanced(self, html: str, domain: str, url: str) -> List[ServicePlan]:
        """Enhanced HTML extraction with improved pattern matching and validation"""
        
        pattern = self.patterns.get(domain, {})
        if not pattern:
            logger.warning(f"No specific pattern for {domain}, using enhanced generic extraction")
            return self._enhanced_generic_extraction(html, domain, url)
        
        plans = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove excluded elements to reduce noise
            exclusion_patterns = pattern.get('exclusion_patterns', [])
            for exclusion in exclusion_patterns:
                for element in soup.select(exclusion):
                    element.decompose()
            
            # Find containers with enhanced selection
            containers = []
            container_selectors = pattern['container_selectors']
            
            for selector in container_selectors:
                found = soup.select(selector)
                containers.extend(found)
            
            # Remove duplicates and filter out tiny containers
            unique_containers = []
            seen_texts = set()
            
            for container in containers:
                container_text = container.get_text(strip=True)
                if len(container_text) > 50 and container_text not in seen_texts:
                    seen_texts.add(container_text)
                    unique_containers.append(container)
            
            logger.info(f"Found {len(unique_containers)} unique plan containers for {domain}")
            
            # Extract plans from containers
            for i, container in enumerate(unique_containers):
                try:
                    plan = self._extract_single_plan_enhanced(container, domain, pattern, url, i)
                    if plan:
                        plans.append(plan)
                except Exception as e:
                    logger.warning(f"Failed to extract plan from container {i}: {e}")
                    continue
            
            # Post-processing: remove duplicates and validate
            unique_plans = self._deduplicate_and_validate_plans(plans)
            
            for plan in unique_plans:
                logger.info(f"✅ Extracted: {plan.name} - {plan.monthly_price} NOK ({plan.confidence:.2f})")
            
            return unique_plans
            
        except Exception as e:
            logger.error(f"Enhanced HTML extraction failed for {domain}: {e}", exc_info=True)
            return []
    
    def _extract_single_plan_enhanced(self, container, domain: str, pattern: Dict, url: str, index: int) -> Optional[ServicePlan]:
        """Extract individual plan with enhanced validation and confidence scoring"""
        
        try:
            confidence_score = 1.0
            
            # Extract plan name with multiple strategies
            name = self._extract_plan_name(container, pattern)
            if not name or len(name) < 3:
                return None
            
            # Extract price with enhanced pattern matching
            price, price_confidence = self._extract_plan_price(container, pattern)
            confidence_score *= price_confidence
            
            # Extract features with intelligent filtering
            features = self._extract_plan_features(container, pattern)
            
            # Extract additional metadata for better categorization
            metadata = self._extract_plan_metadata(container, pattern)
            
            # Calculate final confidence score based on data quality
            if price > 0:
                confidence_score *= 1.0
            else:
                confidence_score *= 0.6
            
            if len(features) > 0:
                confidence_score *= 1.0
            else:
                confidence_score *= 0.8
            
            if len(name) > 10:
                confidence_score *= 1.0
            else:
                confidence_score *= 0.9
            
            # Create plan with comprehensive data
            plan_id = f"{domain.replace('.', '_')}_{hashlib.md5(f'{name}_{price}_{index}'.encode()).hexdigest()[:8]}"
            
            return ServicePlan(
                id=plan_id,
                provider=domain.split('.')[0].title(),
                name=name,
                category=pattern['category'],
                monthly_price=price,
                features=features[:15],  # Limit to 15 most relevant features
                url=url,
                extracted_at=datetime.now().isoformat(),
                confidence=min(confidence_score, 1.0),
                extraction_method="playwright_enhanced",
                data_source="web",
                validation_score=self._calculate_validation_score(name, price, features)
            )
            
        except Exception as e:
            logger.error(f"Enhanced single plan extraction failed: {e}")
            return None
    
    def _extract_plan_name(self, container, pattern: Dict) -> str:
        """Extract plan name using multiple selector strategies"""
        name_selectors = pattern.get('name_selectors', ['h3', '.plan-name'])
        
        for selector in name_selectors:
            try:
                name_elem = container.select_one(selector)
                if name_elem and name_elem.get_text(strip=True):
                    name = name_elem.get_text(strip=True)
                    
                    # Clean up common unwanted text
                    name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
                    name = re.sub(r'(kr|NOK|\$|€|£).*$', '', name, flags=re.IGNORECASE)  # Remove price info
                    name = name.strip()
                    
                    if 3 <= len(name) <= 100:  # Reasonable name length
                        return name
            except Exception:
                continue
        
        return "Unknown Plan"
    
    def _extract_plan_price(self, container, pattern: Dict) -> tuple[float, float]:
        """Extract price with confidence scoring"""
        price = 0.0
        confidence = 0.5  # Default confidence for price extraction
        
        # Try price-specific selectors first
        price_selectors = pattern.get('price_selectors', [])
        for selector in price_selectors:
            try:
                price_elem = container.select_one(selector)
                if price_elem:
                    price_text = price_elem.get_text()
                    extracted_price = self._parse_price_text(price_text, pattern)
                    if extracted_price > 0:
                        return extracted_price, 1.0  # High confidence for dedicated price selectors
            except Exception:
                continue
        
        # Fallback to regex pattern on entire container text
        container_text = container.get_text()
        price_regex = pattern.get('price_regex', r'(\d+)\s*kr')
        
        try:
            price_matches = re.findall(price_regex, container_text, re.IGNORECASE)
            if price_matches:
                # Take the first reasonable price found
                for match in price_matches:
                    try:
                        extracted_price = float(match.replace(',', '.'))
                        if 10 <= extracted_price <= 10000:  # Reasonable price range
                            return extracted_price, 0.8  # Medium confidence for regex extraction
                    except (ValueError, TypeError):
                        continue
        except Exception:
            pass
        
        return price, confidence
    
    def _parse_price_text(self, price_text: str, pattern: Dict) -> float:
        """Parse price text with Norwegian format support"""
        if not price_text:
            return 0.0
        
        # Remove common currency symbols and text
        price_text = re.sub(r'[^\d,.\s]', '', price_text)
        price_text = price_text.strip()
        
        # Handle Norwegian number format (comma as decimal separator)
        price_text = price_text.replace(',', '.')
        
        # Extract first number that looks like a price
        numbers = re.findall(r'\d+\.?\d*', price_text)
        for number in numbers:
            try:
                price = float(number)
                if 10 <= price <= 10000:  # Reasonable price range for Norwegian services
                    return price
            except ValueError:
                continue
        
        return 0.0
    
    def _extract_plan_features(self, container, pattern: Dict) -> List[str]:
        """Extract features with intelligent filtering and deduplication"""
        features = []
        features_selectors = pattern.get('features_selectors', ['li', '.feature'])
        
        for selector in features_selectors:
            try:
                feature_elements = container.select(selector)
                for elem in feature_elements:
                    feature_text = elem.get_text(strip=True)
                    
                    # Filter out unwanted feature text
                    if self._is_valid_feature(feature_text):
                        features.append(feature_text)
                        
            except Exception:
                continue
        
        # Remove duplicates while preserving order
        unique_features = []
        seen_features = set()
        
        for feature in features:
            feature_normalized = feature.lower().strip()
            if feature_normalized not in seen_features and len(feature_normalized) > 3:
                seen_features.add(feature_normalized)
                unique_features.append(feature)
        
        return unique_features[:15]  # Limit to 15 features
    
    def _is_valid_feature(self, feature_text: str) -> bool:
        """Validate if text represents a meaningful feature"""
        if not feature_text or len(feature_text) < 5:
            return False
        
        if len(feature_text) > 200:  # Too long to be a feature
            return False
        
        # Filter out common non-feature text
        invalid_patterns = [
            r'^(les mer|read more|more info|click here|her|there)$',
            r'^(ja|nei|yes|no)$',
            r'^\d+$',  # Just numbers
            r'^[^a-zA-ZæøåÆØÅ]*$',  # No letters
            r'(cookie|privacy|terms|conditions)',
            r'(footer|header|navigation|menu)',
        ]
        
        for pattern in invalid_patterns:
            if re.search(pattern, feature_text, re.IGNORECASE):
                return False
        
        return True
    
    def _extract_plan_metadata(self, container, pattern: Dict) -> Dict[str, Any]:
        """Extract additional metadata for better plan categorization"""
        metadata = {}
        
        try:
            # Look for contract duration
            duration_patterns = [
                r'(\d+)\s*(måned|month|år|year)',
                r'(ingen|no)\s*(binding|commitment)',
                r'(binding|commitment)\s*(\d+)'
            ]
            
            container_text = container.get_text().lower()
            for pattern_regex in duration_patterns:
                match = re.search(pattern_regex, container_text)
                if match:
                    metadata['contract_info'] = match.group(0)
                    break
            
            # Look for special offers or discounts
            offer_patterns = [
                r'(tilbud|offer|rabatt|discount|gratis|free)',
                r'(\d+)%\s*(rabatt|discount|off)',
                r'(første|first)\s*(\d+)\s*(måned|month)'
            ]
            
            for pattern_regex in offer_patterns:
                if re.search(pattern_regex, container_text):
                    metadata['has_offer'] = True
                    break
            
        except Exception as e:
            logger.debug(f"Metadata extraction failed: {e}")
        
        return metadata
    
    def _calculate_validation_score(self, name: str, price: float, features: List[str]) -> float:
        """Calculate validation score based on data completeness and quality"""
        score = 0.0
        
        # Name quality (40% of score)
        if name and name != "Unknown Plan":
            if len(name) > 5:
                score += 0.4
            else:
                score += 0.2
        
        # Price quality (35% of score)
        if price > 0:
            if 50 <= price <= 5000:  # Reasonable price range
                score += 0.35
            else:
                score += 0.2
        
        # Features quality (25% of score)
        if features:
            feature_score = min(len(features) / 10.0, 1.0) * 0.25
            score += feature_score
        
        return min(score, 1.0)
    
    def _deduplicate_and_validate_plans(self, plans: List[ServicePlan]) -> List[ServicePlan]:
        """Remove duplicates and validate plans with enhanced logic"""
        if not plans:
            return []
        
        # Sort by confidence score (highest first)
        plans.sort(key=lambda p: p.confidence, reverse=True)
        
        unique_plans = []
        seen_signatures = set()
        
        for plan in plans:
            # Create signature for deduplication
            signature = self._create_plan_signature(plan)
            
            if signature not in seen_signatures:
                # Additional validation
                if self._validate_plan_quality(plan):
                    seen_signatures.add(signature)
                    unique_plans.append(plan)
        
        return unique_plans
    
    def _create_plan_signature(self, plan: ServicePlan) -> str:
        """Create unique signature for plan deduplication"""
        # Normalize name for comparison
        normalized_name = re.sub(r'[^\w\s]', '', plan.name.lower()).strip()
        normalized_name = re.sub(r'\s+', ' ', normalized_name)
        
        # Create signature based on name and price
        signature = f"{normalized_name}_{plan.monthly_price}"
        return hashlib.md5(signature.encode()).hexdigest()
    
    def _validate_plan_quality(self, plan: ServicePlan) -> bool:
        """Validate plan quality before inclusion in results"""
        # Must have meaningful name
        if not plan.name or plan.name == "Unknown Plan" or len(plan.name) < 3:
            return False
        
        # Must have either price or features
        if plan.monthly_price <= 0 and not plan.features:
            return False
        
        # Confidence threshold
        if plan.confidence < 0.3:
            return False
        
        # Name should not be too generic
        generic_names = ['plan', 'package', 'offer', 'subscription', 'abonnement']
        if plan.name.lower().strip() in generic_names:
            return False
        
        return True
    
    def _enhanced_generic_extraction(self, html: str, domain: str, url: str) -> List[ServicePlan]:
        """Enhanced generic extraction for domains without specific patterns"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            plans = []
            
            # Enhanced selectors for plan detection
            plan_indicators = [
                '[class*="plan"]:not([class*="floor"]):not([class*="plane"])',
                '[class*="price"]:not([class*="comparison"])',
                '[class*="package"]:not([class*="shipping"])',
                '[class*="subscription"]',
                '[class*="abonnement"]',
                '[data-testid*="plan"]',
                '[data-testid*="price"]',
                '.product-card',
                '.service-card'
            ]
            
            containers = []
            for indicator in plan_indicators:
                found_containers = soup.select(indicator)
                containers.extend(found_containers)
            
            # Filter containers by size and content
            valid_containers = []
            for container in containers:
                container_text = container.get_text(strip=True)
                if 50 <= len(container_text) <= 2000:  # Reasonable size
                    valid_containers.append(container)
            
            # Extract from valid containers
            for i, container in enumerate(valid_containers[:10]):  # Limit to 10 for performance
                try:
                    plan = self._extract_generic_plan(container, domain, url, i)
                    if plan:
                        plans.append(plan)
                except Exception as e:
                    logger.debug(f"Generic extraction failed for container {i}: {e}")
                    continue
            
            return plans[:5]  # Limit generic results
            
        except Exception as e:
            logger.error(f"Enhanced generic extraction failed: {e}")
            return []
    
    def _extract_generic_plan(self, container, domain: str, url: str, index: int) -> Optional[ServicePlan]:
        """Extract plan using generic patterns"""
        try:
            # Find name using common heading selectors
            name_elements = container.find_all(['h1', 'h2', 'h3', 'h4', 'h5'])
            name = "Unknown Plan"
            
            for elem in name_elements:
                text = elem.get_text(strip=True)
                if 3 <= len(text) <= 50:
                    name = text
                    break
            
            if name == "Unknown Plan":
                return None
            
            # Try to find price
            price = 0.0
            container_text = container.get_text()
            
            # Norwegian price patterns
            price_patterns = [
                r'(\d+)\s*kr',
                r'(\d+)\s*,-',
                r'kr\s*(\d+)',
                r'(\d+)\s*øre'
            ]
            
            for pattern in price_patterns:
                matches = re.findall(pattern, container_text, re.IGNORECASE)
                if matches:
                    try:
                        price = float(matches[0])
                        if price < 10 and 'øre' in pattern:  # Convert øre to kr
                            price = price / 100
                        break
                    except ValueError:
                        continue
            
            # Extract basic features
            features = []
            feature_elements = container.find_all(['li', 'p'])
            for elem in feature_elements:
                feature_text = elem.get_text(strip=True)
                if 5 <= len(feature_text) <= 100:
                    features.append(feature_text)
            
            plan_id = f"{domain.replace('.', '_')}_generic_{index}"
            
            return ServicePlan(
                id=plan_id,
                provider=domain.split('.')[0].title(),
                name=name,
                category='unknown',
                monthly_price=price,
                features=features[:5],  # Limit features for generic extraction
                url=url,
                extracted_at=datetime.now().isoformat(),
                confidence=0.4,  # Lower confidence for generic extraction
                extraction_method="generic",
                data_source="web",
                validation_score=0.3
            )
            
        except Exception as e:
            logger.debug(f"Generic plan extraction failed: {e}")
            return None
    
    def _enhanced_fallback_extraction(self, url: str, domain: str) -> List[ServicePlan]:
        """Enhanced fallback extraction with session management and retries"""
        try:
            logger.info(f"Starting enhanced fallback extraction for {url}")
            
            # Create session with comprehensive headers
            session = requests.Session()
            
            # Set realistic headers for Norwegian context
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'no,nb;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0'
            })
            
            # Retry mechanism with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = session.get(url, timeout=30)
                    response.raise_for_status()
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' not in content_type:
                        logger.warning(f"Unexpected content type: {content_type}")
                        continue
                    
                    # Check response size
                    if len(response.content) < 1000:
                        logger.warning(f"Response too small: {len(response.content)} bytes")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                    
                    # Extract plans from response
                    plans = self._extract_plans_from_html_enhanced(response.text, domain, url)
                    
                    if plans:
                        # Mark as fallback extraction with adjusted confidence
                        for plan in plans:
                            plan.extraction_method = "fallback_enhanced"
                            plan.confidence = min(plan.confidence * 0.8, 1.0)  # Reduce confidence slightly
                        
                        logger.info(f"Enhanced fallback extraction successful: {len(plans)} plans")
                        return plans
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Fallback attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        raise
            
            return []
            
        except Exception as e:
            logger.error(f"Enhanced fallback extraction failed: {e}")
            return []
    
    def _basic_extraction(self, url: str, domain: str) -> List[ServicePlan]:
        """Basic extraction as last resort with minimal requirements"""
        try:
            logger.info(f"Starting basic extraction for {url}")
            
            # Simple request with minimal headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Very basic plan detection
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
            
            plans = []
            for i, heading in enumerate(headings[:5]):  # Limit to 5
                heading_text = heading.get_text(strip=True)
                
                # Check if heading might be a plan name
                if (3 <= len(heading_text) <= 50 and 
                    any(keyword in heading_text.lower() for keyword in 
                        ['plan', 'pakke', 'abonnement', 'subscription', 'mobile', 'mobil'])):
                    
                    plan_id = f"{domain.replace('.', '_')}_basic_{i}"
                    
                    plan = ServicePlan(
                        id=plan_id,
                        provider=domain.split('.')[0].title(),
                        name=heading_text,
                        category='unknown',
                        monthly_price=0.0,
                        features=[],
                        url=url,
                        extracted_at=datetime.now().isoformat(),
                        confidence=0.2,  # Very low confidence
                        extraction_method="basic",
                        data_source="web",
                        validation_score=0.1
                    )
                    plans.append(plan)
            
            logger.info(f"Basic extraction completed: {len(plans)} plans")
            return plans
            
        except Exception as e:
            logger.error(f"Basic extraction failed: {e}")
            return []
    
    def submit_extraction_task(self, url: str, provider_name: str) -> str:
        """Submit extraction task to background queue with validation"""
        
        # Validate URL
        if not url or not (url.startswith('http://') or url.startswith('https://')):
            raise ValueError("Invalid URL provided")
        
        # Check if we're at capacity
        if self.task_queue.qsize() >= 45:  # Leave some buffer
            raise RuntimeError("Task queue is full, please try again later")
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create task with comprehensive configuration
        task = ExtractionTask(
            task_id=task_id,
            url=url,
            provider_name=provider_name,
            status=ExtractionStatus.PENDING,
            message="Task queued for processing",
            max_retries=3
        )
        
        # Store task in active tasks
        self.active_tasks[task_id] = task
        
        # Update database
        self.db_manager.update_task(task)
        
        # Queue for processing
        try:
            self.task_queue.put(task, timeout=1.0)
            logger.info(f"Submitted extraction task {task_id} for {provider_name}")
            return task_id
        except queue.Full:
            # Remove from active tasks if queue is full
            self.active_tasks.pop(task_id, None)
            raise RuntimeError("Unable to queue task, system is busy")
    
    def get_task_status(self, task_id: str) -> Optional[ExtractionTask]:
        """Get current status of extraction task"""
        return self.active_tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running extraction task"""
        task = self.active_tasks.get(task_id)
        if task and task.status in [ExtractionStatus.PENDING, ExtractionStatus.RUNNING]:
            task.status = ExtractionStatus.CANCELLED
            task.message = "Task cancelled by user"
            self.db_manager.update_task(task)
            logger.info(f"Task {task_id} cancelled")
            return True
        return False
    
    def get_all_plans(self) -> List[Dict[str, Any]]:
        """Get all plans from database with caching"""
        return self.db_manager.get_all_plans()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics with caching for performance"""
        current_time = time.time()
        
        # Use cache if recent (within 30 seconds)
        if (self._stats_cache_time and 
            current_time - self._stats_cache_time < 30 and 
            self._stats_cache):
            return self._stats_cache
        
        # Get fresh statistics
        stats = self.db_manager.get_statistics()
        
        # Add runtime statistics
        stats.update({
            'active_tasks_count': len(self.active_tasks),
            'queue_size': self.task_queue.qsize(),
            'active_browsers': self.browser_manager.get_active_browser_count(),
            'uptime_seconds': int(current_time - getattr(self, '_start_time', current_time)),
            'system_status': 'healthy' if len(self.active_tasks) < 10 else 'busy'
        })
        
        # Cache results
        self._stats_cache = stats
        self._stats_cache_time = current_time
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status for monitoring"""
        try:
            stats = self.get_statistics()
            
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'database': 'healthy' if stats.get('total_plans', 0) >= 0 else 'unhealthy',
                    'task_queue': 'healthy' if self.task_queue.qsize() < 45 else 'overloaded',
                    'browsers': 'healthy' if self.browser_manager.get_active_browser_count() < 3 else 'busy',
                    'workers': 'healthy' if not self._shutdown_event.is_set() else 'shutdown'
                },
                'metrics': {
                    'active_tasks': len(self.active_tasks),
                    'queue_size': self.task_queue.qsize(),
                    'total_plans': stats.get('total_plans', 0),
                    'recent_extractions': stats.get('recent_extractions', 0)
                }
            }
            
            # Overall health assessment
            unhealthy_components = [k for k, v in health_status['components'].items() if v != 'healthy']
            if unhealthy_components:
                health_status['status'] = 'degraded' if len(unhealthy_components) == 1 else 'unhealthy'
                health_status['issues'] = unhealthy_components
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            # Fallback to simple string processing
            return url.replace('https://', '').replace('http://', '').split('/')[0].lower()
    
    def shutdown(self) -> None:
        """Gracefully shutdown the extraction engine with proper cleanup"""
        logger.info("Starting graceful shutdown of extraction engine...")
        
        try:
            # Signal shutdown to all components
            self._shutdown_event.set()
            
            # Cancel all pending tasks
            for task_id, task in self.active_tasks.items():
                if task.status == ExtractionStatus.PENDING:
                    task.status = ExtractionStatus.CANCELLED
                    task.message = "System shutdown"
                    self.db_manager.update_task(task)
            
            # Add shutdown signals to task queue
            for _ in range(3):  # Number of workers
                try:
                    self.task_queue.put(None, timeout=1.0)
                except queue.Full:
                    break
            
            # Shutdown worker pool with timeout
            logger.info("Shutting down worker pool...")
            self.worker_pool.shutdown(wait=True, timeout=15)
            
            # Shutdown async manager
            logger.info("Shutting down async manager...")
            self.async_manager.shutdown()
            
            # Final database cleanup
            self.db_manager.cleanup_old_tasks(days_old=1)
            
            logger.info("Extraction engine shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)


# Global instance management
_extraction_engine = None
_engine_lock = threading.Lock()


def get_extraction_engine() -> ExtractionEngine:
    """Get singleton extraction engine instance with thread safety"""
    global _extraction_engine
    
    if _extraction_engine is None:
        with _engine_lock:
            if _extraction_engine is None:
                _extraction_engine = ExtractionEngine()
                _extraction_engine._start_time = time.time()
    
    return _extraction_engine


def shutdown_extraction_engine():
    """Shutdown the global extraction engine instance"""
    global _extraction_engine
    
    if _extraction_engine is not None:
        with _engine_lock:
            if _extraction_engine is not None:
                _extraction_engine.shutdown()
                _extraction_engine = None


# Context manager for automatic cleanup
class ExtractionEngineContext:
    """Context manager for extraction engine with automatic cleanup"""
    
    def __init__(self):
        self.engine = None
    
    def __enter__(self):
        self.engine = get_extraction_engine()
        return self.engine
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Exception in extraction engine context: {exc_val}")
        # Don't shutdown here as it's a singleton - let application handle shutdown


# Export main classes and functions
__all__ = [
    'ExtractionEngine', 
    'ServicePlan', 
    'ExtractionTask', 
    'ExtractionStatus',
    'get_extraction_engine', 
    'shutdown_extraction_engine',
    'ExtractionEngineContext'
]