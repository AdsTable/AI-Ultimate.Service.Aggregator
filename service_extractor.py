# service_extractor.py
"""
Enhanced Service Extractor - Windows Compatible
Robust web scraping engine for Norwegian service providers with anti-detection capabilities
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
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ServicePlan:
    """Service plan data structure for standardized plan information"""
    id: str
    provider: str
    name: str
    category: str
    monthly_price: float
    features: str  # JSON string for database storage
    url: str
    extracted_at: str
    confidence: float = 1.0


class ServiceExtractor:
    """
    Core extraction engine for Norwegian service providers
    Focus: Reliability, simplicity, Windows compatibility, and cost efficiency
    """
    
    def __init__(self):
        """Initialize the service extractor with proper asyncio configuration"""
        self.db_path = "services.db"
        self.patterns = self._load_extraction_patterns()
        self.session_count = 0
        self._setup_asyncio_policy()
        self._initialize_database()
        print("✅ Service Extractor initialized")
    
    def _setup_asyncio_policy(self) -> None:
        """Setup proper asyncio policy for Windows compatibility"""
        if platform.system() == 'Windows':
            # Use SelectorEventLoop on Windows to support subprocesses
            if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            else:
                # Fallback for older Python versions
                asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    
    def _load_extraction_patterns(self) -> Dict[str, Any]:
        """Load proven extraction patterns for Norwegian service providers"""
        return {
            # Mobile operators - tested patterns
            "telia.no": {
                "category": "mobile",
                "price_regex": r'(\d+)\s*kr',
                "name_selectors": ['h3', '.plan-name', '.subscription-title', '[data-testid*="plan"]'],
                "container_selectors": ['.plan-card', '.subscription-box', '.product-card'],
                "features_selectors": ['li', '.feature', '.benefit', '.included']
            },
            "telenor.no": {
                "category": "mobile",
                "price_regex": r'(\d+)\s*kr',
                "name_selectors": ['h3', '.plan-title', '[data-cy*="plan"]'],
                "container_selectors": ['.product-card', '.plan-container', '.mobile-plan'],
                "features_selectors": ['li', '.feature-list li', '.benefits li']
            },
            "ice.no": {
                "category": "mobile",
                "price_regex": r'(\d+)\s*kr',
                "name_selectors": ['h3', '.plan-name', '.title'],
                "container_selectors": ['.plan-box', '.offer-card'],
                "features_selectors": ['li', '.features li']
            },
            "talkmore.no": {
                "category": "mobile",
                "price_regex": r'(\d+)\s*kr',
                "name_selectors": ['h3', '.plan-title', '.price-title'],
                "container_selectors": ['.plan-card', '.price-box', '.subscription'],
                "features_selectors": ['li', '.feature']
            },
            # Electricity providers
            "fortum.no": {
                "category": "electricity",
                "price_regex": r'(\d+[,\.]\d+)\s*øre',
                "name_selectors": ['h3', '.plan-title', '.product-name'],
                "container_selectors": ['.price-plan', '.electricity-plan'],
                "features_selectors": ['li', '.feature']
            },
            "hafslund.no": {
                "category": "electricity",
                "price_regex": r'(\d+[,\.]\d+)\s*øre',
                "name_selectors": ['h3', '.plan-name'],
                "container_selectors": ['.plan-card', '.price-card'],
                "features_selectors": ['li', '.benefits li']
            }
        }
    
    def _initialize_database(self) -> None:
        """Create SQLite database with optimized schema and indexes"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS plans (
            id TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            monthly_price REAL,
            features TEXT,
            url TEXT,
            extracted_at TEXT,
            confidence REAL DEFAULT 1.0
        )
        """)
        
        # Create performance indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_provider ON plans(provider)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON plans(category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_price ON plans(monthly_price)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_extracted ON plans(extracted_at)")
        conn.commit()
        conn.close()
        print("✅ Database initialized")
    
    def extract_from_provider_sync(self, url: str) -> List[ServicePlan]:
        """
        Synchronous wrapper for provider extraction.
        Creates isolated event loop to avoid conflicts with Streamlit.
        """
        try:
            # Use ThreadPoolExecutor to isolate async operations from Streamlit
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._run_async_extraction, url)
                return future.result(timeout=120)  # 2 minute timeout
                
        except Exception as e:
            print(f"❌ Sync extraction failed: {str(e)}")
            return self._fallback_extraction(url)
    
    def _run_async_extraction(self, url: str) -> List[ServicePlan]:
        """Run async extraction in isolated event loop to prevent conflicts"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                return loop.run_until_complete(self._extract_from_provider_async(url))
            finally:
                loop.close()
                
        except Exception as e:
            print(f"❌ Async extraction failed: {str(e)}")
            return self._fallback_extraction(url)
    
    async def _extract_from_provider_async(self, url: str) -> List[ServicePlan]:
        """Main async extraction method with comprehensive error handling"""
        domain = self._extract_domain(url)
        print(f"\n🎯 Starting extraction from: {domain}")
        
        try:
            # Try Playwright extraction first (most reliable)
            plans = await self._playwright_extraction(url, domain)
            
            if plans:
                self._store_plans_in_database(plans)
                print(f"✅ Successfully extracted {len(plans)} plans from {domain}")
                self.session_count += 1
                return plans
            else:
                print(f"⚠️ Playwright extraction returned no plans for {domain}")
                return self._fallback_extraction(url)
                
        except Exception as e:
            print(f"❌ Playwright extraction failed for {domain}: {str(e)}")
            return self._fallback_extraction(url)
    
    async def _playwright_extraction(self, url: str, domain: str) -> List[ServicePlan]:
        """Playwright-based extraction with Windows compatibility and stealth measures"""
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                # Enhanced browser configuration for Windows compatibility
                browser = await p.chromium.launch(
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
                        '--single-process'  # Critical for Windows subprocess compatibility
                    ]
                )
                
                # Create stealth context with realistic browser fingerprint
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent=self._get_random_user_agent(),
                    extra_http_headers=self._get_stealth_headers(),
                    ignore_https_errors=True
                )
                
                page = await context.new_page()
                
                try:
                    # Apply comprehensive stealth measures
                    await self._apply_stealth_measures(page)
                    
                    # Navigate with retry mechanism
                    plans = await self._navigate_and_extract(page, url, domain)
                    
                    return plans
                    
                except Exception as e:
                    print(f"❌ Page extraction failed for {domain}: {str(e)}")
                    return []
                
                finally:
                    # Cleanup resources properly
                    try:
                        await page.close()
                        await context.close()
                        await browser.close()
                    except:
                        pass  # Ignore cleanup errors
        
        except Exception as e:
            print(f"❌ Browser initialization failed: {str(e)}")
            return []
    
    async def _apply_stealth_measures(self, page) -> None:
        """Apply comprehensive anti-detection measures"""
        # Remove webdriver detection signatures
        await page.add_init_script("""
            // Remove webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            // Mock realistic plugins array
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    {name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer'},
                    {name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai'},
                    {name: 'Native Client', filename: 'internal-nacl-plugin'}
                ],
            });
            
            // Mock realistic languages array
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en', 'no'],
            });
            
            // Override permissions API
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        """)
        
        # Block tracking and analytics scripts to improve performance and stealth
        await page.route("**/*", lambda route: (
            route.abort() if any(domain in route.request.url for domain in [
                'google-analytics.com', 'googletagmanager.com', 'facebook.com/tr',
                'hotjar.com', 'fullstory.com', 'recaptcha.net', 'doubleclick.net'
            ]) else route.continue_()
        ))
    
    async def _navigate_and_extract(self, page, url: str, domain: str) -> List[ServicePlan]:
        """Navigate to page and extract data with retry mechanism and human simulation"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                print(f"🌐 Navigating to {url} (attempt {attempt + 1})")
                
                # Navigate with appropriate timeout
                response = await page.goto(
                    url, 
                    wait_until='domcontentloaded', 
                    timeout=45000
                )
                
                # Check for HTTP errors
                if response and response.status >= 400:
                    print(f"⚠️ HTTP {response.status} received")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                
                # Handle cookie consent dialogs
                await self._handle_cookie_consent(page)
                
                # Wait for dynamic content to load
                await asyncio.sleep(3)
                
                # Simulate realistic human browsing behavior
                await self._simulate_human_behavior(page)
                
                # Extract page content
                html_content = await page.content()
                plans = self._extract_plans_from_html(html_content, domain)
                
                return plans
                
            except Exception as e:
                print(f"❌ Navigation attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return []
        
        return []
    
    async def _handle_cookie_consent(self, page) -> bool:
        """Handle cookie consent dialogs with Norwegian-specific patterns"""
        # Common Norwegian cookie consent patterns
        cookie_patterns = [
            'button:has-text("Godta alle")',  # Accept all (Norwegian)
            'button:has-text("Godta")',       # Accept (Norwegian)
            'button:has-text("Accept all")',  # Accept all (English)
            'button:has-text("Accept")',      # Accept (English)
            'button:has-text("OK")',          # OK
            '.cookie-accept',
            '.accept-all-cookies',
            '[data-testid*="accept"]',
            '[data-cy*="accept"]',
            '[id*="CookieConsent"] button',
            '.cookie-banner button'
        ]
        
        for pattern in cookie_patterns:
            try:
                await page.click(pattern, timeout=2000)
                await asyncio.sleep(1)
                print("✅ Cookie consent handled")
                return True
            except:
                continue
        
        print("ℹ️ No cookie consent dialog found")
        return False
    
    async def _simulate_human_behavior(self, page) -> None:
        """Simulate realistic human browsing behavior to avoid detection"""
        try:
            # Random initial delay
            await asyncio.sleep(random.uniform(1.0, 3.0))
            
            # Random mouse movements
            for _ in range(random.randint(3, 6)):
                x = random.randint(100, 1800)
                y = random.randint(100, 900)
                await page.mouse.move(x, y)
                await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # Natural scrolling behavior
            for _ in range(random.randint(2, 4)):
                scroll_distance = random.randint(100, 500)
                await page.mouse.wheel(0, scroll_distance)
                await asyncio.sleep(random.uniform(0.8, 2.0))
            
            # Occasional random clicks on non-functional elements
            if random.random() < 0.3:
                try:
                    elements = await page.query_selector_all('div, span')
                    if elements:
                        element = random.choice(elements[:5])
                        await element.click()
                        await asyncio.sleep(random.uniform(0.5, 1.5))
                except:
                    pass  # Ignore click failures
                    
        except Exception as e:
            print(f"⚠️ Behavior simulation failed: {str(e)}")
    
    def _fallback_extraction(self, url: str) -> List[ServicePlan]:
        """Fallback extraction using requests + BeautifulSoup when Playwright fails"""
        try:
            print(f"🔄 Attempting fallback extraction for {url}")
            
            headers = self._get_stealth_headers()
            headers['User-Agent'] = self._get_random_user_agent()
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            domain = self._extract_domain(url)
            plans = self._extract_plans_from_html(response.text, domain)
            
            if plans:
                self._store_plans_in_database(plans)
                print(f"✅ Fallback extraction successful: {len(plans)} plans")
            
            return plans
            
        except Exception as e:
            print(f"❌ Fallback extraction failed: {str(e)}")
            return []
    
    def _extract_plans_from_html(self, html: str, domain: str) -> List[ServicePlan]:
        """Extract service plans using domain-specific patterns with robust error handling"""
        if domain not in self.patterns:
            print(f"⚠️ No extraction pattern for {domain}, trying generic extraction")
            return self._generic_plan_extraction(html, domain)
        
        pattern_config = self.patterns[domain]
        plans = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find plan containers using configured selectors
            containers = []
            for selector in pattern_config['container_selectors']:
                found_containers = soup.select(selector)
                containers.extend(found_containers)
            
            print(f"🔍 Found {len(containers)} potential plan containers")
            
            for container in containers:
                plan = self._extract_single_plan(container, domain, pattern_config)
                if plan:
                    plans.append(plan)
            
            # Remove duplicate plans based on name
            seen_names = set()
            unique_plans = []
            for plan in plans:
                if plan.name not in seen_names:
                    seen_names.add(plan.name)
                    unique_plans.append(plan)
                    print(f"✅ Extracted: {plan.name} - {plan.monthly_price} kr")
            
            return unique_plans
            
        except Exception as e:
            print(f"❌ Pattern extraction failed for {domain}: {e}")
            return []
    
    def _extract_single_plan(self, container, domain: str, pattern_config: Dict) -> Optional[ServicePlan]:
        """Extract data from a single plan container using pattern configuration"""
        try:
            # Extract plan name using configured selectors
            name = "Unknown Plan"
            for name_selector in pattern_config['name_selectors']:
                name_elem = container.select_one(name_selector)
                if name_elem and name_elem.get_text().strip():
                    name = name_elem.get_text().strip()
                    break
            
            # Extract price using regex pattern
            price = 0.0
            container_text = container.get_text()
            price_match = re.search(pattern_config['price_regex'], container_text)
            if price_match:
                try:
                    price_str = price_match.group(1).replace(',', '.')
                    price = float(price_str)
                except (ValueError, IndexError):
                    price = 0.0
            
            # Extract features using configured selectors
            features = []
            for feature_selector in pattern_config['features_selectors']:
                feature_elements = container.select(feature_selector)
                for elem in feature_elements:
                    feature_text = elem.get_text().strip()
                    if feature_text and 5 < len(feature_text) < 200:  # Reasonable feature length
                        features.append(feature_text)
            
            # Create plan if we have meaningful data
            if name != "Unknown Plan" and (price > 0 or features):
                plan_id = f"{domain.replace('.', '_')}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
                
                return ServicePlan(
                    id=plan_id,
                    provider=domain.split('.')[0].title(),
                    name=name,
                    category=pattern_config['category'],
                    monthly_price=price,
                    features=json.dumps(features[:10], ensure_ascii=False),  # Limit to 10 features
                    url=f"https://{domain}",
                    extracted_at=datetime.now().isoformat(),
                    confidence=0.9  # High confidence for pattern-based extraction
                )
            
            return None
            
        except Exception as e:
            print(f"❌ Single plan extraction failed: {str(e)}")
            return None
    
    def _generic_plan_extraction(self, html: str, domain: str) -> List[ServicePlan]:
        """Generic extraction for domains without specific patterns"""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            plans = []
            
            # Look for common plan indicators
            plan_indicators = [
                '[class*="plan"]', '[class*="price"]', '[class*="package"]',
                '[data-testid*="plan"]', '[data-testid*="price"]'
            ]
            
            for indicator in plan_indicators:
                containers = soup.select(indicator)
                for container in containers[:5]:  # Limit to prevent too many false positives
                    # Try to extract basic information
                    name_elem = container.find(['h1', 'h2', 'h3', 'h4'])
                    if name_elem and name_elem.get_text().strip():
                        name = name_elem.get_text().strip()
                        
                        plan_id = f"{domain.replace('.', '_')}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
                        
                        plan = ServicePlan(
                            id=plan_id,
                            provider=domain.split('.')[0].title(),
                            name=name,
                            category='unknown',
                            monthly_price=0.0,
                            features=json.dumps([]),
                            url=f"https://{domain}",
                            extracted_at=datetime.now().isoformat(),
                            confidence=0.3  # Low confidence for generic extraction
                        )
                        plans.append(plan)
            
            return plans[:5]  # Limit to 5 plans for generic extraction
            
        except Exception as e:
            print(f"❌ Generic extraction failed: {str(e)}")
            return []
    
    def _store_plans_in_database(self, plans: List[ServicePlan]) -> None:
        """Store extracted plans in SQLite database with error handling"""
        if not plans:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            for plan in plans:
                conn.execute("""
                INSERT OR REPLACE INTO plans 
                (id, provider, name, category, monthly_price, features, url, extracted_at, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    plan.id, plan.provider, plan.name, plan.category,
                    plan.monthly_price, plan.features, plan.url,
                    plan.extracted_at, plan.confidence
                ))
            
            conn.commit()
            conn.close()
            print(f"💾 Stored {len(plans)} plans in database")
            
        except Exception as e:
            print(f"❌ Database storage failed: {str(e)}")
    
    def search_plans(self, category: str = None, max_price: float = None, 
                    provider: str = None, limit: int = 50) -> List[Dict]:
        """Search plans in database with flexible filtering options"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            
            query = "SELECT * FROM plans WHERE 1=1"
            params = []
            
            if category:
                query += " AND category = ?"
                params.append(category)
            
            if max_price:
                query += " AND monthly_price <= ?"
                params.append(max_price)
            
            if provider:
                query += " AND provider LIKE ?"
                params.append(f"%{provider}%")
            
            query += " ORDER BY monthly_price ASC LIMIT ?"
            params.append(limit)
            
            results = conn.execute(query, params).fetchall()
            conn.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            print(f"❌ Database search failed: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive extraction and database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Total plans
            total_plans = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
            
            # Plans by category with average prices
            category_stats = conn.execute("""
                SELECT category, COUNT(*) as count, ROUND(AVG(monthly_price), 2) as avg_price
                FROM plans WHERE monthly_price > 0
                GROUP BY category
            """).fetchall()
            
            # Plans by provider
            provider_stats = conn.execute("""
                SELECT provider, COUNT(*) as count
                FROM plans GROUP BY provider ORDER BY count DESC
            """).fetchall()
            
            # Recent extractions (last 24 hours)
            recent_extractions = conn.execute("""
                SELECT COUNT(*) as count FROM plans
                WHERE extracted_at > datetime('now', '-24 hours')
            """).fetchone()[0]
            
            conn.close()
            
            return {
                "total_plans": total_plans,
                "categories": {row[0]: {"count": row[1], "avg_price": row[2]} for row in category_stats},
                "providers": {row[0]: row[1] for row in provider_stats},
                "recent_extractions": recent_extractions,
                "extraction_sessions": self.session_count
            }
            
        except Exception as e:
            print(f"❌ Statistics generation failed: {str(e)}")
            return {
                "total_plans": 0,
                "categories": {},
                "providers": {},
                "recent_extractions": 0,
                "extraction_sessions": self.session_count
            }
    
    def get_all_plans(self) -> List[Dict[str, Any]]:
        """Get all plans from database for UI display"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("""
                SELECT id, provider, name, category, monthly_price, features, url, extracted_at, confidence
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
                    'confidence': row[8]
                })
            
            conn.close()
            return plans
            
        except Exception as e:
            print(f"❌ Database read failed: {str(e)}")
            return []
    
    def _extract_domain(self, url: str) -> str:
        """Extract clean domain from URL"""
        return url.replace('https://', '').replace('http://', '').split('/')[0]
    
    def _get_random_user_agent(self) -> str:
        """Get a random realistic user agent for stealth"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        return random.choice(user_agents)
    
    def _get_stealth_headers(self) -> Dict[str, str]:
        """Get stealth HTTP headers that mimic real browser requests"""
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


# Thread-safe wrapper functions for Streamlit integration
def run_extraction_in_streamlit(url: str) -> List[ServicePlan]:
    """
    Thread-safe extraction wrapper for Streamlit applications.
    Prevents event loop conflicts by isolating async operations.
    """
    extractor = ServiceExtractor()
    return extractor.extract_from_provider_sync(url)


def get_extractor_instance() -> ServiceExtractor:
    """Get a new extractor instance for use in applications"""
    return ServiceExtractor()