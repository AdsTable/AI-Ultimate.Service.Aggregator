# agents/content_extraction_agent.py
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
import json
import re

from services.ai_async_client import AIAsyncClient
from playwright.async_api import async_playwright

@dataclass
class ExtractionResult:
    """Comprehensive extraction result with AI insights"""
    provider_name: str
    website_url: str
    services: List[Dict[str, Any]]
    pricing_info: Dict[str, Any]
    contact_details: Dict[str, Any]
    business_info: Dict[str, Any]
    competitive_analysis: Dict[str, Any]
    extraction_confidence: float
    raw_data: Dict[str, Any]

class IntelligentContentExtractor:
    """Advanced AI-powered content extraction with multi-modal understanding"""
    
    def __init__(self, ai_client: AIAsyncClient):
        self.ai_client = ai_client
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def extract_comprehensive_data(self, website_url: str, service_category: str) -> ExtractionResult:
        """
        Extract comprehensive data using multi-stage AI analysis
        """
        
        # Stage 1: Website structure analysis
        structure_analysis = await self._analyze_website_structure(website_url)
        
        # Stage 2: Content extraction with AI guidance
        content_data = await self._extract_guided_content(website_url, structure_analysis, service_category)
        
        # Stage 3: AI-powered data synthesis
        synthesized_data = await self._synthesize_extracted_data(content_data, service_category)
        
        # Stage 4: Competitive positioning analysis
        competitive_analysis = await self._analyze_competitive_positioning(synthesized_data, service_category)
        
        return ExtractionResult(
            provider_name=synthesized_data.get('provider_name', 'Unknown'),
            website_url=website_url,
            services=synthesized_data.get('services', []),
            pricing_info=synthesized_data.get('pricing', {}),
            contact_details=synthesized_data.get('contact', {}),
            business_info=synthesized_data.get('business_info', {}),
            competitive_analysis=competitive_analysis,
            extraction_confidence=synthesized_data.get('confidence', 0.0),
            raw_data=content_data
        )
    
    async def _analyze_website_structure(self, url: str) -> Dict[str, Any]:
        """AI-powered website structure analysis"""
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Navigate and wait for content
                await page.goto(url, wait_until='networkidle')
                
                # Get page structure information
                structure_info = await page.evaluate("""
                () => {
                    return {
                        title: document.title,
                        headings: Array.from(document.querySelectorAll('h1,h2,h3')).map(h => h.textContent),
                        nav_links: Array.from(document.querySelectorAll('nav a, .nav a, .menu a')).map(a => ({
                            text: a.textContent,
                            href: a.href
                        })),
                        forms: Array.from(document.querySelectorAll('form')).length,
                        has_pricing_section: !!document.querySelector('[class*="price"], [id*="price"], [class*="plan"], [id*="plan"]'),
                        has_contact_section: !!document.querySelector('[class*="contact"], [id*="contact"], [href*="contact"]'),
                        meta_description: document.querySelector('meta[name="description"]')?.content || '',
                        structured_data: Array.from(document.querySelectorAll('script[type="application/ld+json"]')).map(s => s.textContent)
                    }
                }
                """)
                
                await browser.close()
                
                # AI analysis of structure
                structure_prompt = f"""
                Analyze this website structure and recommend optimal data extraction strategy:
                
                URL: {url}
                Structure Info: {json.dumps(structure_info, indent=2)}
                
                Provide extraction recommendations:
                1. Key sections to focus on
                2. CSS selectors for important content
                3. Extraction priority order
                4. Potential challenges and solutions
                
                Return JSON with specific extraction guidance.
                """
                
                response = await self.ai_client.ask(structure_prompt, provider="ollama")
                ai_guidance = json.loads(response.content)
                
                return {
                    'structure_info': structure_info,
                    'ai_guidance': ai_guidance
                }
                
        except Exception as e:
            logging.error(f"Structure analysis failed for {url}: {e}")
            return {}
    
    async def _extract_guided_content(self, url: str, structure_analysis: Dict, service_category: str) -> Dict[str, Any]:
        """Extract content using AI guidance"""
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                page = await context.new_page()
                
                # Navigate to website
                await page.goto(url, wait_until='networkidle')
                
                # Extract text content using AI-guided selectors
                ai_guidance = structure_analysis.get('ai_guidance', {})
                priority_selectors = ai_guidance.get('priority_selectors', [])
                
                extracted_content = {}
                
                # Extract using recommended selectors
                for selector_info in priority_selectors:
                    try:
                        selector = selector_info.get('selector')
                        content_type = selector_info.get('type')
                        
                        elements = await page.query_selector_all(selector)
                        content = []
                        
                        for element in elements:
                            text = await element.inner_text()
                            content.append(text)
                        
                        extracted_content[content_type] = content
                        
                    except Exception as e:
                        logging.error(f"Selector extraction failed for {selector}: {e}")
                
                # Fallback: extract all text content
                if not extracted_content:
                    full_text = await page.evaluate("() => document.body.innerText")
                    extracted_content['full_text'] = full_text
                
                await browser.close()
                
                return extracted_content
                
        except Exception as e:
            logging.error(f"Guided extraction failed for {url}: {e}")
            return {}
    
    async def _synthesize_extracted_data(self, content_data: Dict, service_category: str) -> Dict[str, Any]:
        """AI-powered data synthesis and structuring"""
        
        synthesis_prompt = f"""
        Synthesize and structure this extracted website content for a {service_category} provider:
        
        Extracted Content: {json.dumps(content_data, indent=2)[:10000]}
        
        Extract and structure:
        1. **Provider Information**:
           - Company name
           - Business description
           - Years in operation
           - Company size indicators
           - Certifications/licenses
        
        2. **Services & Products**:
           - Service offerings with details
           - Pricing information
           - Package/plan comparisons
           - Special offers or promotions
        
        3. **Contact & Location**:
           - Phone numbers
           - Email addresses
           - Physical addresses
           - Service areas
           - Customer service hours
        
        4. **Competitive Positioning**:
           - Unique value propositions
           - Target customer segments
           - Key differentiators
           - Market positioning claims
        
        Return well-structured JSON with confidence scores for each extracted field.
        """
        
        try:
            response = await self.ai_client.ask(synthesis_prompt, provider="ollama")
            synthesized_data = json.loads(response.content)
            
            return synthesized_data
            
        except Exception as e:
            logging.error(f"Data synthesis failed: {e}")
            return {}