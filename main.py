# main.py
"""
Service Aggregator - Production Ready System
A simple yet powerful service comparison platform
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
import streamlit as st
import pandas as pd

@dataclass
class ServicePlan:
    """Service plan data structure"""
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
    Focus: Reliability, simplicity, and cost efficiency
    """
    
    def __init__(self):
        self.db_path = "services.db"
        self.patterns = self._load_extraction_patterns()
        self.session_count = 0
        self._initialize_database()
        print("✅ Service Extractor initialized")
    
    def _load_extraction_patterns(self) -> Dict[str, Any]:
        """Load proven extraction patterns for Norwegian providers"""
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
    
    def _initialize_database(self):
        """Create SQLite database with optimized schema"""
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
            confidence REAL DEFAULT 1.0,
            
            -- Indexes for performance
            UNIQUE(id)
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
    
    async def _apply_stealth_measures(self, page):
        """Apply anti-detection measures"""
        # Remove webdriver detection
        await page.add_init_script("""
        // Remove webdriver property
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        
        // Mock plugins
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3].map(() => ({name: 'Chrome PDF Plugin'}))
        });
        
        // Mock languages
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en', 'no']
        });
        """)
        
        # Block tracking and analytics
        await page.route("**/*", lambda route: (
            route.abort() if any(domain in route.request.url for domain in [
                'google-analytics.com', 'googletagmanager.com', 'facebook.com/tr',
                'hotjar.com', 'fullstory.com', 'recaptcha.net'
            ]) else route.continue_()
        ))
    
    async def _simulate_human_behavior(self, page):
        """Simulate realistic human browsing behavior"""
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
        
        # Occasional random clicks (non-functional)
        if random.random() < 0.3:
            try:
                elements = await page.query_selector_all('div, span')
                if elements:
                    element = random.choice(elements[:5])
                    await element.click()
                    await asyncio.sleep(random.uniform(0.5, 1.5))
            except:
                pass  # Ignore click failures
    
    async def _handle_cookie_consent(self, page) -> bool:
        """Handle cookie consent dialogs"""
        # Common Norwegian cookie consent patterns
        cookie_patterns = [
            'button:has-text("Godta alle")',
            'button:has-text("Godta")',
            'button:has-text("Accept all")', 
            'button:has-text("Accept")',
            'button:has-text("OK")',
            '.cookie-accept',
            '.accept-all-cookies',
            '[data-testid*="accept"]',
            '[data-cy*="accept"]'
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
    
    def _extract_plans_from_html(self, html: str, domain: str) -> List[ServicePlan]:
        """Extract service plans using domain-specific patterns"""
        
        if domain not in self.patterns:
            print(f"⚠️ No extraction pattern for {domain}")
            return []
        
        pattern_config = self.patterns[domain]
        plans = []
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find plan containers
            containers = []
            for selector in pattern_config['container_selectors']:
                found_containers = soup.select(selector)
                containers.extend(found_containers)
            
            print(f"🔍 Found {len(containers)} potential plan containers")
            
            for container in containers:
                # Extract plan name
                name = "Unknown Plan"
                for name_selector in pattern_config['name_selectors']:
                    name_elem = container.select_one(name_selector)
                    if name_elem and name_elem.get_text().strip():
                        name = name_elem.get_text().strip()
                        break
                
                # Extract price using regex
                price = 0.0
                container_text = container.get_text()
                import re
                price_match = re.search(pattern_config['price_regex'], container_text)
                if price_match:
                    try:
                        price_str = price_match.group(1).replace(',', '.')
                        price = float(price_str)
                    except (ValueError, IndexError):
                        price = 0.0
                
                # Extract features
                features = []
                for feature_selector in pattern_config['features_selectors']:
                    feature_elements = container.select(feature_selector)
                    for elem in feature_elements:
                        feature_text = elem.get_text().strip()
                        if feature_text and len(feature_text) < 200:  # Reasonable feature length
                            features.append(feature_text)
                
                # Create plan if we have meaningful data
                if name != "Unknown Plan" and (price > 0 or features):
                    plan = ServicePlan(
                        id=f"{domain.replace('.', '_')}_{hashlib.md5(name.encode()).hexdigest()[:8]}",
                        provider=domain.split('.')[0].title(),
                        name=name,
                        category=pattern_config['category'],
                        monthly_price=price,
                        features=json.dumps(features, ensure_ascii=False),
                        url=f"https://{domain}",
                        extracted_at=datetime.now().isoformat(),
                        confidence=0.9  # High confidence for pattern-based extraction
                    )
                    plans.append(plan)
                    print(f"✅ Extracted: {name} - {price} kr")
            
        except Exception as e:
            print(f"❌ Pattern extraction failed for {domain}: {e}")
        
        return plans
    
    async def extract_from_provider(self, url: str) -> List[ServicePlan]:
        """Main extraction method for a single provider"""
        
        domain = url.replace('https://', '').replace('http://', '').split('/')[0]
        print(f"\n🎯 Starting extraction from: {domain}")
        
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                # Launch browser with stealth configuration
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-blink-features=AutomationControlled',
                        '--disable-extensions'
                    ]
                )
                
                # Create stealth context
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    extra_http_headers={
                        'Accept-Language': 'en-US,en;q=0.9,no;q=0.8',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                    }
                )
                
                page = await context.new_page()
                
                try:
                    # Apply stealth measures
                    await self._apply_stealth_measures(page)
                    
                    # Navigate to provider
                    print(f"🌐 Navigating to {url}")
                    await page.goto(url, wait_until='networkidle', timeout=30000)
                    
                    # Handle cookie consent
                    await self._handle_cookie_consent(page)
                    
                    # Simulate human behavior
                    await self._simulate_human_behavior(page)
                    
                    # Get page content
                    html_content = await page.content()
                    
                    # Extract plans using patterns
                    plans = self._extract_plans_from_html(html_content, domain)
                    
                    # Store successful extractions
                    if plans:
                        self._store_plans_in_database(plans)
                        print(f"✅ Successfully extracted {len(plans)} plans from {domain}")
                    else:
                        print(f"⚠️ No plans extracted from {domain}")
                    
                    self.session_count += 1
                    return plans
                    
                except Exception as e:
                    print(f"❌ Extraction failed for {domain}: {str(e)}")
                    return []
                
                finally:
                    await browser.close()
        
        except Exception as e:
            print(f"❌ Browser setup failed: {str(e)}")
            return []
    
    def _store_plans_in_database(self, plans: List[ServicePlan]):
        """Store extracted plans in SQLite database"""
        
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
    
    def search_plans(self, category: str = None, max_price: float = None, 
                    provider: str = None, limit: int = 50) -> List[Dict]:
        """Search plans in database with filters"""
        
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction and database statistics"""
        
        conn = sqlite3.connect(self.db_path)
        
        # Total plans
        total_plans = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
        
        # Plans by category
        category_stats = conn.execute("""
        SELECT category, COUNT(*) as count, ROUND(AVG(monthly_price), 2) as avg_price
        FROM plans 
        WHERE monthly_price > 0
        GROUP BY category
        """).fetchall()
        
        # Plans by provider
        provider_stats = conn.execute("""
        SELECT provider, COUNT(*) as count
        FROM plans 
        GROUP BY provider
        ORDER BY count DESC
        """).fetchall()
        
        # Recent extractions
        recent_extractions = conn.execute("""
        SELECT COUNT(*) as count
        FROM plans 
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

def create_admin_interface():
    """Create Streamlit-based admin interface"""
    
    # Page configuration
    st.set_page_config(
        page_title="Service Aggregator Admin",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better appearance
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Service Aggregator - Admin Dashboard</h1>
        <p>Norwegian Service Provider Comparison Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize extractor
    if 'extractor' not in st.session_state:
        with st.spinner("Initializing Service Extractor..."):
            st.session_state.extractor = ServiceExtractor()
    
    extractor = st.session_state.extractor
    
    # Sidebar controls
    st.sidebar.header("🎛️ Control Panel")
    
    # Manual extraction section
    st.sidebar.subheader("Manual Extraction")
    
    # Predefined providers
    provider_options = {
        "Telia Mobile": "https://www.telia.no/privat/mobil/abonnement",
        "Telenor Mobile": "https://www.telenor.no/privat/mobil/abonnement",
        "Ice Mobile": "https://www.ice.no/mobil/abonnement",
        "Fortum Electricity": "https://www.fortum.no/privat/strom",
        "Hafslund Electricity": "https://www.hafslund.no/privat/strom"
    }
    
    selected_provider = st.sidebar.selectbox(
        "Select Provider",
        list(provider_options.keys())
    )
    
    if st.sidebar.button("🚀 Extract Selected Provider"):
        with st.spinner(f"Extracting from {selected_provider}..."):
            url = provider_options[selected_provider]
            
            # Show progress
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            try:
                status_text.text("Starting extraction...")
                progress_bar.progress(25)
                
                plans = asyncio.run(extractor.extract_from_provider(url))
                progress_bar.progress(100)
                
                if plans:
                    st.sidebar.success(f"✅ Extracted {len(plans)} plans!")
                    status_text.text(f"Successfully extracted {len(plans)} plans")
                else:
                    st.sidebar.warning("⚠️ No plans found")
                    status_text.text("No plans extracted")
                
            except Exception as e:
                st.sidebar.error(f"❌ Extraction failed: {str(e)}")
                status_text.text("Extraction failed")
            
            progress_bar.empty()
    
    # Custom URL extraction
    st.sidebar.subheader("Custom URL")
    custom_url = st.sidebar.text_input("Enter URL")
    
    if st.sidebar.button("🔍 Extract Custom URL"):
        if custom_url:
            with st.spinner("Extracting from custom URL..."):
                try:
                    plans = asyncio.run(extractor.extract_from_provider(custom_url))
                    if plans:
                        st.sidebar.success(f"✅ Extracted {len(plans)} plans!")
                    else:
                        st.sidebar.warning("⚠️ No plans found")
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")
        else:
            st.sidebar.error("Please enter a URL")
    
    # Bulk extraction
    st.sidebar.subheader("Bulk Operations")
    
    if st.sidebar.button("🔄 Extract All Known Providers"):
        all_urls = list(provider_options.values())
        total_plans = 0
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        for i, (name, url) in enumerate(provider_options.items()):
            status_text.text(f"Extracting from {name}...")
            
            try:
                plans = asyncio.run(extractor.extract_from_provider(url))
                total_plans += len(plans)
                progress_bar.progress((i + 1) / len(provider_options))
                
            except Exception as e:
                st.sidebar.error(f"Failed to extract from {name}: {str(e)}")
        
        status_text.text(f"Completed! Total: {total_plans} plans")
        st.sidebar.success(f"🎉 Bulk extraction complete! Total: {total_plans} plans")
    
    # Main dashboard area
    col1, col2, col3, col4 = st.columns(4)
    
    # Get current statistics
    stats = extractor.get_statistics()
    
    # Display key metrics
    with col1:
        st.metric(
            label="📊 Total Plans",
            value=stats['total_plans'],
            delta=f"+{stats['recent_extractions']} today"
        )
    
    with col2:
        st.metric(
            label="🏢 Providers",
            value=len(stats['providers']),
            delta=None
        )
    
    with col3:
        st.metric(
            label="📂 Categories",
            value=len(stats['categories']),
            delta=None
        )
    
    with col4:
        st.metric(
            label="🔄 Sessions",
            value=stats['extraction_sessions'],
            delta=None
        )
    
    # Search and filter interface
    st.subheader("🔍 Search Service Plans")
    
    search_col1, search_col2, search_col3, search_col4 = st.columns(4)
    
    with search_col1:
        category_options = ["All"] + list(stats['categories'].keys())
        selected_category = st.selectbox("Category", category_options)
    
    with search_col2:
        max_price = st.number_input("Max Price (kr)", min_value=0, max_value=5000, value=1000)
    
    with search_col3:
        provider_filter = st.text_input("Provider")
    
    with search_col4:
        results_limit = st.number_input("Results Limit", min_value=10, max_value=100, value=50)
    
    # Execute search
    search_params = {}
    if selected_category != "All":
        search_params['category'] = selected_category
    if max_price > 0:
        search_params['max_price'] = max_price
    if provider_filter:
        search_params['provider'] = provider_filter
    
    search_params['limit'] = results_limit
    
    # Get search results
    results = extractor.search_plans(**search_params)
    
    # Display results
    if results:
        st.subheader(f"📋 Found {len(results)} Plans")
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(results)
        
        # Process features column for display
        if 'features' in df.columns:
            def format_features(features_json):
                try:
                    features = json.loads(features_json) if features_json else []
                    return ', '.join(features[:3]) + ('...' if len(features) > 3 else '')
                except:
                    return 'N/A'
            
            df['features_display'] = df['features'].apply(format_features)
            df = df.drop('features', axis=1)
        
        # Format extracted_at for readability
        if 'extracted_at' in df.columns:
            df['extracted_at'] = pd.to_datetime(df['extracted_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Display with formatting
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "monthly_price": st.column_config.NumberColumn(
                    "Price (kr)",
                    format="%.0f kr"
                ),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0,
                    max_value=1
                )
            }
        )
        
        # Export options
        st.subheader("📤 Export Options")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # CSV export
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name=f"service_plans_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with export_col2:
            # JSON export
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Download as JSON",
                data=json_data,
                file_name=f"service_plans_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    else:
        st.info("🔍 No plans found with current filters. Try adjusting your search criteria.")
    
    # Statistics and analytics
    if stats['total_plans'] > 0:
        st.subheader("📈 Analytics Dashboard")
        
        analytics_col1, analytics_col2 = st.columns(2)
        
        with analytics_col1:
            # Category distribution
            if stats['categories']:
                st.write("**Plans by Category**")
                category_df = pd.DataFrame([
                    {"Category": cat, "Count": data['count'], "Avg Price": data['avg_price']}
                    for cat, data in stats['categories'].items()
                ])
                st.bar_chart(category_df.set_index('Category')['Count'])
        
        with analytics_col2:
            # Provider distribution
            if stats['providers']:
                st.write("**Plans by Provider**")
                provider_df = pd.DataFrame([
                    {"Provider": provider, "Count": count}
                    for provider, count in list(stats['providers'].items())[:10]
                ])
                st.bar_chart(provider_df.set_index('Provider'))
    
    # System status
    st.subheader("🔧 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.write("**Database Status**")
        if stats['total_plans'] > 0:
            st.success("✅ Active")
        else:
            st.warning("⚠️ Empty")
    
    with status_col2:
        st.write("**Last Activity**")
        if stats['recent_extractions'] > 0:
            st.success(f"✅ {stats['recent_extractions']} extractions today")
        else:
            st.info("ℹ️ No recent activity")
    
    with status_col3:
        st.write("**Extraction Patterns**")
        pattern_count = len(extractor.patterns)
        st.info(f"📝 {pattern_count} providers configured")

# Main application entry point
if __name__ == "__main__":
    create_admin_interface()