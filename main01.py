# main.py
"""
Ultimate Service Aggregator - Production Ready
Radical simplification focused on immediate deployment and low costs
"""
import asyncio
import sqlite3
import json
import time
import random
import hashlib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from playwright.async_api import async_playwright, Page
import gspread
from google.oauth2.service_account import Credentials
import streamlit as st
import pandas as pd

@dataclass
class ServicePlan:
    """Simplified service plan structure"""
    id: str
    provider: str
    name: str
    category: str
    monthly_price: float
    features: str  # JSON string for simplicity
    url: str
    extracted_at: str
    confidence: float = 1.0

class MinimalExtractor:
    """
    Ultra-simplified extractor focused on cost efficiency and reliability
    Key principles:
    - No AI for simple tasks
    - Pattern-based extraction with minimal learning
    - Human-like behavior without ML complexity
    - Maximum stealth with minimum code
    """
    
    def __init__(self):
        self.db_path = "services.db"
        self.patterns = self._load_simple_patterns()
        self.session_count = 0
        self._init_database()
    
    def _load_simple_patterns(self) -> Dict[str, Any]:
        """Load simple, reliable extraction patterns"""
        return {
            # Norwegian mobile operators - static but reliable patterns
            "telia.no": {
                "category": "mobile",
                "price_pattern": r'(\d+)\s*kr',
                "name_selectors": ['h3', '.plan-name', '.subscription-title'],
                "container_selectors": ['.plan-card', '.subscription-box']
            },
            "telenor.no": {
                "category": "mobile", 
                "price_pattern": r'(\d+)\s*kr',
                "name_selectors": ['h3', '.plan-title'],
                "container_selectors": ['.product-card', '.plan-container']
            },
            "ice.no": {
                "category": "mobile",
                "price_pattern": r'(\d+)\s*kr', 
                "name_selectors": ['h3', '.plan-name'],
                "container_selectors": ['.plan-box']
            },
            # Electricity providers
            "fortum.no": {
                "category": "electricity",
                "price_pattern": r'(\d+[,\.]\d+)\s*øre',
                "name_selectors": ['h3', '.plan-title'],
                "container_selectors": ['.price-plan']
            }
        }
    
    def _init_database(self):
        """Initialize simple SQLite database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS plans (
            id TEXT PRIMARY KEY,
            provider TEXT,
            name TEXT,
            category TEXT,
            monthly_price REAL,
            features TEXT,
            url TEXT,
            extracted_at TEXT,
            confidence REAL
        )
        """)
        
        # Simple indexes for performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_provider ON plans(provider)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON plans(category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_price ON plans(monthly_price)")
        conn.commit()
        conn.close()
    
    async def _human_like_session(self, page: Page):
        """Minimal but effective human behavior simulation"""
        
        # Random delays between actions
        await asyncio.sleep(random.uniform(1.0, 3.0))
        
        # Random mouse movements (3-5 movements)
        for _ in range(random.randint(3, 5)):
            x = random.randint(100, 1800)
            y = random.randint(100, 900)
            await page.mouse.move(x, y)
            await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Random scrolling
        for _ in range(random.randint(2, 4)):
            scroll_y = random.randint(100, 500)
            await page.mouse.wheel(0, scroll_y)
            await asyncio.sleep(random.uniform(0.5, 1.5))
    
    async def _bypass_simple_protections(self, page: Page):
        """Simple but effective anti-detection"""
        
        # Remove webdriver properties
        await page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
        Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
        """)
        
        # Block unnecessary requests to speed up and reduce detection
        await page.route("**/*", lambda route: (
            route.abort() if any(domain in route.request.url for domain in [
                'google-analytics', 'facebook.com/tr', 'hotjar.com', 'googletagmanager'
            ]) else route.continue_()
        ))
    
    async def _handle_cookies_simple(self, page: Page) -> bool:
        """Simple cookie handling without AI"""
        
        # Common Norwegian cookie acceptance patterns
        cookie_selectors = [
            'button:has-text("Godta")',
            'button:has-text("Accept")', 
            'button:has-text("OK")',
            '.cookie-accept',
            '.accept-all',
            '[data-testid*="accept"]'
        ]
        
        for selector in cookie_selectors:
            try:
                await page.click(selector, timeout=2000)
                await asyncio.sleep(1)
                return True
            except:
                continue
        
        return False
    
    def _extract_with_patterns(self, html: str, domain: str) -> List[ServicePlan]:
        """Pattern-based extraction without AI - fast and reliable"""
        
        if domain not in self.patterns:
            return []
        
        pattern_config = self.patterns[domain]
        plans = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find plan containers
            containers = []
            for selector in pattern_config['container_selectors']:
                containers.extend(soup.select(selector))
            
            for container in containers:
                # Extract name
                name = "Unknown Plan"
                for name_selector in pattern_config['name_selectors']:
                    name_elem = container.select_one(name_selector)
                    if name_elem:
                        name = name_elem.get_text().strip()
                        break
                
                # Extract price
                price = 0.0
                price_text = container.get_text()
                import re
                price_match = re.search(pattern_config['price_pattern'], price_text)
                if price_match:
                    try:
                        price = float(price_match.group(1).replace(',', '.'))
                    except:
                        price = 0.0
                
                # Simple feature extraction
                features = []
                for elem in container.select('li, .feature, .benefit'):
                    feature_text = elem.get_text().strip()
                    if feature_text and len(feature_text) < 100:
                        features.append(feature_text)
                
                if name != "Unknown Plan" and (price > 0 or features):
                    plan = ServicePlan(
                        id=f"{domain}_{hashlib.md5(name.encode()).hexdigest()[:8]}",
                        provider=domain.split('.')[0].title(),
                        name=name,
                        category=pattern_config['category'],
                        monthly_price=price,
                        features=json.dumps(features),
                        url=f"https://{domain}",
                        extracted_at=datetime.now().isoformat(),
                        confidence=0.8  # Pattern-based confidence
                    )
                    plans.append(plan)
            
        except Exception as e:
            print(f"Pattern extraction failed for {domain}: {e}")
        
        return plans
    
    async def extract_from_url(self, url: str) -> List[ServicePlan]:
        """Main extraction method - simple and reliable"""
        
        domain = url.replace('https://', '').replace('http://', '').split('/')[0]
        
        print(f"🔍 Extracting from: {domain}")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            page = await context.new_page()
            
            try:
                # Apply basic anti-detection
                await self._bypass_simple_protections(page)
                
                # Navigate to page
                await page.goto(url, wait_until='networkidle', timeout=30000)
                
                # Handle cookies
                await self._handle_cookies_simple(page)
                
                # Human-like behavior
                await self._human_like_session(page)
                
                # Get page content
                html = await page.content()
                
                # Extract using patterns
                plans = self._extract_with_patterns(html, domain)
                
                # Store in database
                if plans:
                    self._store_plans(plans)
                
                print(f"✅ Extracted {len(plans)} plans from {domain}")
                self.session_count += 1
                
                return plans
                
            except Exception as e:
                print(f"❌ Extraction failed for {domain}: {e}")
                return []
            
            finally:
                await browser.close()
    
    def _store_plans(self, plans: List[ServicePlan]):
        """Store plans in SQLite database"""
        
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
    
    def search_plans(self, category: str = None, max_price: float = None, 
                    provider: str = None, limit: int = 20) -> List[Dict]:
        """Simple database search"""
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        
        conn = sqlite3.connect(self.db_path)
        
        total = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
        
        categories = conn.execute("""
        SELECT category, COUNT(*) as count, AVG(monthly_price) as avg_price
        FROM plans GROUP BY category
        """).fetchall()
        
        providers = conn.execute("""
        SELECT provider, COUNT(*) as count
        FROM plans GROUP BY provider
        ORDER BY count DESC
        """).fetchall()
        
        conn.close()
        
        return {
            "total_plans": total,
            "categories": dict(categories),
            "providers": dict(providers),
            "extraction_sessions": self.session_count
        }

class GoogleSheetsIntegration:
    """Simple Google Sheets integration for data analysis"""
    
    def __init__(self, credentials_file: str, sheet_name: str):
        self.credentials_file = credentials_file
        self.sheet_name = sheet_name
        self.client = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Sheets API"""
        try:
            creds = Credentials.from_service_account_file(
                self.credentials_file,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            self.client = gspread.authorize(creds)
        except Exception as e:
            print(f"Google Sheets authentication failed: {e}")
    
    def export_plans(self, plans: List[Dict]):
        """Export plans to Google Sheets"""
        
        if not self.client:
            return False
        
        try:
            # Open or create spreadsheet
            try:
                sheet = self.client.open(self.sheet_name).sheet1
            except:
                sheet = self.client.create(self.sheet_name).sheet1
            
            # Clear existing data
            sheet.clear()
            
            # Prepare data
            if plans:
                headers = list(plans[0].keys())
                values = [headers]
                
                for plan in plans:
                    row = [str(plan.get(header, '')) for header in headers]
                    values.append(row)
                
                # Update sheet
                sheet.update('A1', values)
                
                print(f"✅ Exported {len(plans)} plans to Google Sheets")
                return True
        
        except Exception as e:
            print(f"Google Sheets export failed: {e}")
        
        return False

def create_streamlit_admin():
    """Create Streamlit admin interface"""
    
    st.set_page_config(
        page_title="Service Aggregator Admin",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Service Aggregator - Admin Panel")
    
    # Initialize extractor
    if 'extractor' not in st.session_state:
        st.session_state.extractor = MinimalExtractor()
    
    extractor = st.session_state.extractor
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # Manual extraction
    st.sidebar.subheader("Manual Extraction")
    url_input = st.sidebar.text_input("URL to extract")
    
    if st.sidebar.button("Extract Now"):
        if url_input:
            with st.spinner("Extracting..."):
                plans = asyncio.run(extractor.extract_from_url(url_input))
                st.success(f"Extracted {len(plans)} plans")
        else:
            st.error("Please enter a URL")
    
    # Automatic extraction for known providers
    st.sidebar.subheader("Auto Extraction")
    if st.sidebar.button("Extract All Known Providers"):
        known_urls = [
            "https://www.telia.no/privat/mobil/abonnement",
            "https://www.telenor.no/privat/mobil/abonnement", 
            "https://www.ice.no/mobil/abonnement",
            "https://www.fortum.no/privat/strom"
        ]
        
        total_plans = 0
        progress_bar = st.sidebar.progress(0)
        
        for i, url in enumerate(known_urls):
            with st.spinner(f"Extracting from {url}..."):
                plans = asyncio.run(extractor.extract_from_url(url))
                total_plans += len(plans)
                progress_bar.progress((i + 1) / len(known_urls))
        
        st.sidebar.success(f"Total extracted: {total_plans} plans")
    
    # Main dashboard
    col1, col2, col3 = st.columns(3)
    
    stats = extractor.get_stats()
    
    with col1:
        st.metric("Total Plans", stats['total_plans'])
    
    with col2:
        st.metric("Providers", len(stats['providers']))
    
    with col3:
        st.metric("Extraction Sessions", stats['extraction_sessions'])
    
    # Search and filter
    st.subheader("Search Plans")
    
    search_col1, search_col2, search_col3 = st.columns(3)
    
    with search_col1:
        category_filter = st.selectbox("Category", 
                                     ["All"] + list(stats['categories'].keys()))
    
    with search_col2:
        max_price_filter = st.number_input("Max Price", min_value=0, value=1000)
    
    with search_col3:
        provider_filter = st.text_input("Provider")
    
    # Execute search
    search_params = {}
    if category_filter != "All":
        search_params['category'] = category_filter
    if max_price_filter > 0:
        search_params['max_price'] = max_price_filter
    if provider_filter:
        search_params['provider'] = provider_filter
    
    plans = extractor.search_plans(**search_params)
    
    # Display results
    if plans:
        st.subheader(f"Found {len(plans)} plans")
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(plans)
        if 'features' in df.columns:
            df['features'] = df['features'].apply(lambda x: ', '.join(json.loads(x)) if x else '')
        
        st.dataframe(df, use_container_width=True)
        
        # Google Sheets export
        if st.button("Export to Google Sheets"):
            sheets_integration = GoogleSheetsIntegration(
                "credentials.json",  # Service account credentials
                "Service Plans Data"
            )
            success = sheets_integration.export_plans(plans)
            if success:
                st.success("Data exported to Google Sheets!")
            else:
                st.error("Export failed - check credentials")
    
    else:
        st.info("No plans found with current filters")
    
    # Statistics charts
    if stats['categories']:
        st.subheader("Statistics")
        
        # Category distribution
        category_df = pd.DataFrame(list(stats['categories'].items()), 
                                 columns=['Category', 'Count'])
        st.bar_chart(category_df.set_index('Category'))

# Simple API server (alternative to Streamlit)
def create_simple_api():
    """Create simple Flask API for integration"""
    from flask import Flask, jsonify, request
    
    app = Flask(__name__)
    extractor = MinimalExtractor()
    
    @app.route('/api/extract', methods=['POST'])
    def extract_endpoint():
        url = request.json.get('url')
        if not url:
            return jsonify({'error': 'URL required'}), 400
        
        try:
            plans = asyncio.run(extractor.extract_from_url(url))
            return jsonify({
                'success': True,
                'plans_extracted': len(plans),
                'plans': [asdict(plan) for plan in plans]
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/search', methods=['GET'])
    def search_endpoint():
        category = request.args.get('category')
        max_price = request.args.get('max_price', type=float)
        provider = request.args.get('provider')
        limit = request.args.get('limit', 20, type=int)
        
        plans = extractor.search_plans(category, max_price, provider, limit)
        return jsonify({
            'plans': plans,
            'count': len(plans)
        })
    
    @app.route('/api/stats', methods=['GET'])
    def stats_endpoint():
        return jsonify(extractor.get_stats())
    
    return app

# Deployment script
def deploy_service():
    """Simple deployment script"""
    
    import subprocess
    import sys
    
    # Install requirements
    requirements = [
        'streamlit',
        'playwright', 
        'beautifulsoup4',
        'pandas',
        'gspread',
        'google-auth',
        'flask'
    ]
    
    for req in requirements:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
    
    # Install Playwright browser
    subprocess.check_call([sys.executable, '-m', 'playwright', 'install', 'chromium'])
    
    print("✅ Service deployed successfully!")
    print("Run: streamlit run admin_interface.py")

if __name__ == "__main__":
    # Quick test
    async def test_extraction():
        extractor = MinimalExtractor()
        
        test_urls = [
            "https://www.telia.no/privat/mobil/abonnement",
            "https://www.telenor.no/privat/mobil/abonnement"
        ]
        
        for url in test_urls:
            plans = await extractor.extract_from_url(url)
            print(f"Extracted {len(plans)} plans from {url}")
        
        # Show stats
        stats = extractor.get_stats()
        print(f"Total plans in database: {stats['total_plans']}")
    
    # Run test
    asyncio.run(test_extraction())