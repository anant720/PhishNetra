"""
URL analysis pipeline for PhishNetra
Performs safe, read-only, and analytical checks on URLs to detect phishing and malicious intent.

MANDATORY FEATURES:
- Domain structure analysis (typosquatting, lookalike detection)
- HTTPS & certificate verification
- Domain age & reputation signals
- Safe redirection analysis (HEAD requests only)
- Content semantic fingerprinting (metadata, forms, brand impersonation)
- Model-based URL risk scoring and category mapping
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlparse, urlunparse
from datetime import datetime, timedelta
import httpx
import ssl
import socket
import certifi
import asyncio
from collections import defaultdict

from ..core.logging import LoggerMixin
from ..core.config import settings
from ..models.decision_fusion import ThreatCategory

# Comprehensive URL patterns - Well-tested regex for URL detection
URL_PATTERN = re.compile(
    r'https?://(?:[-\w.]+)+(?:[:\d]+)?(?:/[^\s]*)?',
    re.IGNORECASE
)

# Domain mention pattern (e.g., "visit example.com" or "go to bank-site.net")
DOMAIN_MENTION_PATTERN = re.compile(
    r'\b(?:visit|go to|check|access|open|click|link|website|site|login to|connect to)\s+([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*(?:\.[a-zA-Z]{2,}))',
    re.IGNORECASE
)

# Direct domain pattern (e.g., "example.com" or "bank-secure.net")
DIRECT_DOMAIN_PATTERN = re.compile(
    r'\b([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*(?:\.[a-zA-Z]{2,}))\b',
    re.IGNORECASE
)

# QR code reference pattern
QR_REFERENCE_PATTERN = re.compile(
    r'\b(?:scan|qr|qrcode|code)\s*(?:this|the|above|below|here)\b',
    re.IGNORECASE
)

# Known URL shortener domains (comprehensive list)
SHORTLINK_DOMAINS = {
    "bit.ly", "tinyurl.com", "ow.ly", "goo.gl", "t.co", "short.link",
    "rebrand.ly", "cutt.ly", "is.gd", "v.gd", "shorturl.at", "tiny.cc",
    "buff.ly", "bit.do", "mcaf.ee", "adf.ly", "shorte.st", "bc.vc",
    "ouo.io", "link.tl", "clk.sh", "clicky.me", "short.am", "tr.im"
}

# Common legitimate TLDs
COMMON_TLDS = {"com", "org", "net", "edu", "gov", "io", "co", "uk", "ca", "au", "de", "fr"}

# Suspicious TLDs (often used for phishing)
SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "download", "stream"}

# Known legitimate brands (for typosquatting detection)
KNOWN_BRANDS = {
    "google", "microsoft", "amazon", "apple", "facebook", "twitter", "linkedin",
    "github", "wikipedia", "paypal", "ebay", "netflix", "spotify", "youtube",
    "instagram", "whatsapp", "telegram", "discord", "reddit", "tumblr",
    "sbi", "hdfc", "icici", "axis", "kotak", "pnb", "canara", "union",
    "sbi.com", "hdfcbank.com", "icicibank.com", "axisbank.com", "kotak.com",
    "paytm", "phonepe", "gpay", "amazonpay", "razorpay", "stripe"
}

class URLAnalyzer(LoggerMixin):
    """
    Comprehensive URL analyzer for phishing detection.
    
    Performs safe, read-only analysis:
    - Domain structure analysis (typosquatting, lookalike, TLD analysis)
    - HTTPS & SSL certificate verification
    - Domain age & reputation signals
    - Safe redirection analysis (HEAD requests only)
    - Content semantic fingerprinting (metadata, forms, brand impersonation)
    - Model-based risk scoring and category mapping
    """

    def __init__(self):
        self.logger.info("Initializing Enhanced URL Analyzer")
        
        # HTTP client with safe defaults (no execution, read-only)
        self.client = httpx.AsyncClient(
            verify=certifi.where(),
            timeout=settings.url_analysis_timeout_seconds,
            follow_redirects=False,  # We'll handle redirects manually for analysis
            max_redirects=0  # No automatic redirects
        )
        
        # Known legitimate domains (expanded)
        self.known_legitimate_domains = {
            "google.com", "microsoft.com", "amazon.com", "apple.com", "facebook.com",
            "twitter.com", "linkedin.com", "github.com", "wikipedia.org",
            "paypal.com", "ebay.com", "netflix.com", "spotify.com", "youtube.com",
            "instagram.com", "whatsapp.com", "telegram.org", "discord.com",
            "sbi.co.in", "hdfcbank.com", "icicibank.com", "axisbank.com",
            "paytm.com", "phonepe.com", "razorpay.com"
        }
        
        # Suspicious domain patterns
        self.known_suspicious_patterns = [
            r"login-\w+\.(com|net|org|info)",
            r"\w+-verify\.(com|net|org|info)",
            r"secure-\w+\.(com|net|org|info)",
            r"\w+-update\.(com|net|org|info)",
            r"\w+-bank\.(com|net|org|info)",
            r"update-\w+\.(com|net|org|info)",
            r"verify-\w+\.(com|net|org|info)"
        ]
        
        # Brand name variations for typosquatting detection
        self.brand_variations = {
            "google": ["go0gle", "g00gle", "goog1e", "goggle", "googel"],
            "microsoft": ["microsft", "micr0soft", "microsoftt"],
            "amazon": ["amaz0n", "amazom", "amazn"],
            "paypal": ["paypall", "paypa1", "paypai"],
            "facebook": ["faceb00k", "facebok", "facebookk"]
        }
        
        # Domain age cache (simulated - in production would use WHOIS)
        self.domain_age_cache = {}
        
        # SSL context for certificate checking
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    async def analyze_url(self, url: str) -> Dict[str, Any]:
        """
        Comprehensive URL analysis pipeline.

        Performs all mandatory checks:
        1. Domain structure analysis
        2. HTTPS & certificate verification
        3. Domain age & reputation
        4. Safe redirection analysis
        5. Content semantic fingerprinting

        Returns:
            Complete URL analysis with risk score, verdict, signals, and reasoning
        """
        self.logger.info(f"Starting comprehensive analysis for URL: {url}")

        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        analysis_results = {
            "url": url,
            "risk_score": 0.0,
            "verdict": "Likely Legitimate",
            "signals": [],
            "details": {},
            "reasoning": []
        }

        try:
            # 1. DOMAIN STRUCTURE ANALYSIS
        domain_analysis = self._analyze_domain_structure(domain)
        analysis_results["details"]["domain_structure"] = domain_analysis

            domain_risk = 0.0
            if domain_analysis["typosquatting_risk"]:
                domain_risk += 35.0
                analysis_results["signals"].append("typosquatting_detected")
                analysis_results["reasoning"].append(f"Domain appears to be typosquatting: {domain_analysis.get('similar_brand', 'unknown brand')}")

            if domain_analysis["lookalike_risk"]:
                domain_risk += 30.0
                analysis_results["signals"].append("lookalike_domain")
                analysis_results["reasoning"].append("Domain matches suspicious pattern")

            if domain_analysis["excessive_subdomains"]:
                domain_risk += 25.0
                analysis_results["signals"].append("excessive_subdomains")
                analysis_results["reasoning"].append(f"Domain has {len(domain.split('.'))} parts, suggesting potential subdomain abuse")

            if domain_analysis["unusual_tld"]:
                domain_risk += 20.0
                analysis_results["signals"].append("unusual_tld")
                analysis_results["reasoning"].append(f"Unusual TLD '{domain_analysis['tld']}' for this brand")

            analysis_results["risk_score"] += domain_risk

        # Check for shortened links
        if domain in SHORTLINK_DOMAINS:
                analysis_results["risk_score"] += 25.0
            analysis_results["signals"].append("shortened_url")
                analysis_results["reasoning"].append(f"URL uses known shortener: {domain}")
            analysis_results["verdict"] = "Suspicious"

            # 2. HTTPS & CERTIFICATE SIGNALS
            https_signals = await self._check_https_and_certificate(url, domain)
        analysis_results["details"]["https_certificate"] = https_signals

        if not https_signals["uses_https"]:
                analysis_results["risk_score"] += 30.0
            analysis_results["signals"].append("no_https")
                analysis_results["reasoning"].append("URL does not use HTTPS encryption")

        if https_signals["suspicious_certificate"]:
                analysis_results["risk_score"] += 35.0
            analysis_results["signals"].append("suspicious_certificate")
                analysis_results["reasoning"].append(f"Certificate issues detected: {https_signals.get('certificate_issues', 'unknown')}")

            if https_signals.get("new_certificate"):
                analysis_results["risk_score"] += 15.0
                analysis_results["signals"].append("new_certificate")
                analysis_results["reasoning"].append("Certificate was recently issued (potential new domain)")

            # 3. DOMAIN AGE & REPUTATION SIGNALS
            domain_reputation = await self._check_domain_age_and_reputation(domain)
        analysis_results["details"]["domain_reputation"] = domain_reputation

        if domain_reputation["new_domain_risk"]:
                analysis_results["risk_score"] += 25.0
            analysis_results["signals"].append("new_domain")
                analysis_results["reasoning"].append(f"Domain appears to be new or recently created (age: {domain_reputation.get('estimated_age_days', 'unknown')} days)")

        if domain_reputation["suspicious_reputation"]:
                analysis_results["risk_score"] += 40.0
            analysis_results["signals"].append("suspicious_reputation")
                analysis_results["reasoning"].append("Domain has suspicious reputation indicators")

            # 4. REDIRECTION BEHAVIOR (Safe HEAD requests only)
            redirection_behavior = await self._analyze_redirections(url)
        analysis_results["details"]["redirection_behavior"] = redirection_behavior

            if redirection_behavior["excessive_redirects"]:
                analysis_results["risk_score"] += 30.0
                analysis_results["signals"].append("excessive_redirects")
                analysis_results["reasoning"].append(f"URL has {redirection_behavior['redirect_count']} redirects (suspicious chain)")

            if redirection_behavior["mismatched_final_domain"]:
                analysis_results["risk_score"] += 40.0
                analysis_results["signals"].append("mismatched_final_domain")
                analysis_results["reasoning"].append(f"URL redirects to different domain: {redirection_behavior.get('final_domain', 'unknown')}")

            if redirection_behavior.get("redirect_chain_detected"):
                analysis_results["risk_score"] += 20.0
                analysis_results["signals"].append("redirect_chain")
                analysis_results["reasoning"].append("Multiple redirects detected in chain")

            # 5. CONTENT SEMANTIC FINGERPRINTING
        content_fingerprinting = await self._fetch_content_metadata(url)
        analysis_results["details"]["content_fingerprinting"] = content_fingerprinting

        if content_fingerprinting["login_form_detected"]:
                analysis_results["risk_score"] += 45.0
            analysis_results["signals"].append("credential_form_detected")
                analysis_results["reasoning"].append("Page contains login/credential form (potential credential harvesting)")

        if content_fingerprinting["brand_impersonation_detected"]:
                analysis_results["risk_score"] += 50.0
            analysis_results["signals"].append("brand_impersonation")
                analysis_results["reasoning"].append(f"Page appears to impersonate brand: {content_fingerprinting.get('impersonated_brand', 'unknown')}")

            if content_fingerprinting.get("payment_gateway_detected"):
                analysis_results["risk_score"] += 35.0
                analysis_results["signals"].append("payment_gateway_detected")
                analysis_results["reasoning"].append("Page contains payment form (potential financial manipulation)")

            # Generate reasoning summary
            if not analysis_results["reasoning"]:
                analysis_results["reasoning"].append("No significant risk indicators detected")

            # Determine verdict based on risk score
            if analysis_results["risk_score"] >= 70:
            analysis_results["verdict"] = "High Risk"
            elif analysis_results["risk_score"] >= 40:
            analysis_results["verdict"] = "Suspicious"
            else:
                analysis_results["verdict"] = "Likely Legitimate"
        
        # Cap risk score at 100
        analysis_results["risk_score"] = min(analysis_results["risk_score"], 100.0)

            # Add contribution to final scam score
            analysis_results["contribution_to_scam_score"] = min(analysis_results["risk_score"] * 0.3, 30.0)  # Max 30 points contribution

            self.logger.info(
                f"URL analysis completed: {url} | Verdict: {analysis_results['verdict']} | "
                f"Risk: {analysis_results['risk_score']:.1f} | Signals: {len(analysis_results['signals'])}"
            )

        except Exception as e:
            self.logger.error(f"Error during URL analysis for {url}: {e}")
            analysis_results["risk_score"] = 50.0  # Default suspicious if analysis fails
            analysis_results["verdict"] = "Suspicious"
            analysis_results["signals"].append("analysis_error")
            analysis_results["reasoning"].append(f"Analysis encountered error: {str(e)}")

        return analysis_results

    def _analyze_domain_structure(self, domain: str) -> Dict[str, Any]:
        """
        Comprehensive domain structure analysis.
        
        Detects:
        - Typosquatting (character substitutions, insertions, deletions)
        - Lookalike domains (homoglyphs, similar patterns)
        - Excessive subdomains
        - Unusual TLDs for known brands
        """
        parts = domain.split('.')
        base_domain = ".".join(parts[-2:]) if len(parts) >= 2 else domain
        tld = parts[-1].lower() if parts else ""
        domain_name = parts[-2].lower() if len(parts) >= 2 else base_domain.split('.')[0] if '.' in base_domain else base_domain
        
        typosquatting_risk = False
        lookalike_risk = False
        excessive_subdomains = False
        unusual_tld = False
        similar_brand = None
        
        # 1. TYPOSQUATTING DETECTION
        # Check against known legitimate domains
        for known_domain in self.known_legitimate_domains:
            known_base = ".".join(known_domain.split('.')[-2:]) if '.' in known_domain else known_domain
            known_name = known_domain.split('.')[0] if '.' in known_domain else known_domain
            
            # Check base domain similarity
            if base_domain != known_base:
                distance = self._calculate_levenshtein_distance(base_domain, known_base)
                if distance <= 2 and len(base_domain) > 3:  # Small edit distance
                    typosquatting_risk = True
                    similar_brand = known_domain
                    break
                
                # Check domain name similarity (without TLD)
                name_distance = self._calculate_levenshtein_distance(domain_name, known_name)
                if name_distance <= 2 and len(domain_name) > 3:
                    typosquatting_risk = True
                    similar_brand = known_domain
                    break
        
        # Check brand variations
        for brand, variations in self.brand_variations.items():
            if brand in domain_name or any(var in domain_name for var in variations):
                if base_domain not in self.known_legitimate_domains:
                typosquatting_risk = True
                    similar_brand = brand
                break
        
        # 2. LOOKALIKE DOMAIN DETECTION
        # Check against suspicious patterns
        for pattern in self.known_suspicious_patterns:
            if re.search(pattern, domain, re.IGNORECASE):
                lookalike_risk = True
                break

        # Check for homoglyph-like substitutions (common phishing technique)
        homoglyph_patterns = [
            r'[0-9]+[a-z]+',  # Numbers mixed with letters (e.g., g00gle)
            r'[a-z]+[0-9]+[a-z]+',  # Letters-numbers-letters
        ]
        for pattern in homoglyph_patterns:
            if re.search(pattern, domain_name):
                # Check if it's similar to a known brand
                for brand in KNOWN_BRANDS:
                    if brand.lower() in domain_name.replace('0', 'o').replace('1', 'i').replace('5', 's'):
                        lookalike_risk = True
                        similar_brand = brand
                        break
        
        # 3. EXCESSIVE SUBDOMAINS
        # Legitimate domains can have subdomains, but excessive nesting is suspicious
        if len(parts) > 4:  # More than 4 parts (e.g., a.b.c.d.com)
            excessive_subdomains = True
        elif len(parts) == 4 and base_domain not in self.known_legitimate_domains:
            # Check for subdomain abuse pattern (e.g., login.microsoft.com.malicious.com)
            if any(part in ['login', 'secure', 'verify', 'update', 'account'] for part in parts[:-2]):
            excessive_subdomains = True
        
        # 4. UNUSUAL TLD DETECTION
        # Check if domain name matches a known brand but uses unusual TLD
        for brand in KNOWN_BRANDS:
            if brand.lower() in domain_name:
                # Known brands typically use .com, .org, .net, country-specific TLDs
                if tld in SUSPICIOUS_TLDS:
                    unusual_tld = True
                    break
                # Check if it's a known brand domain but with wrong TLD
                expected_tlds = ["com", "org", "net", "co", "io", "in", "uk", "ca"]
                if base_domain not in self.known_legitimate_domains and tld not in expected_tlds:
                unusual_tld = True
                    break

        return {
            "typosquatting_risk": typosquatting_risk,
            "lookalike_risk": lookalike_risk,
            "excessive_subdomains": excessive_subdomains,
            "unusual_tld": unusual_tld,
            "base_domain": base_domain,
            "tld": tld,
            "domain_name": domain_name,
            "parts_count": len(parts),
            "similar_brand": similar_brand
        }

    async def _check_https_and_certificate(self, url: str, domain: str) -> Dict[str, Any]:
        """
        Check HTTPS usage and SSL certificate details.
        
        Performs safe, read-only certificate inspection:
        - Verifies HTTPS is used
        - Checks certificate validity
        - Detects suspicious certificate patterns
        - Identifies newly issued certificates
        """
        uses_https = url.startswith('https://')
        suspicious_certificate = False
        new_certificate = False
        certificate_issues = []
        
        if not uses_https:
            return {
                "uses_https": False,
                "suspicious_certificate": False,
                "new_certificate": False,
                "certificate_issues": ["No HTTPS"]
            }
        
        try:
            # Try to get certificate information
            try:
                # Create SSL connection to check certificate
                context = ssl.create_default_context()
                with socket.create_connection((domain, 443), timeout=5) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        cert = ssock.getpeercert()
                        
                        # Check certificate validity
                        if cert:
                            # Check certificate expiration
                            not_after = cert.get('notAfter')
                            if not_after:
                                try:
                                    expire_date = datetime.strptime(not_after, '%b %d %H:%M:%S %Y %Z')
                                    days_until_expiry = (expire_date - datetime.now()).days
                                    
                                    if days_until_expiry < 30:
                                        suspicious_certificate = True
                                        certificate_issues.append("Certificate expiring soon")
                                    
                                    # Check if certificate is very new (issued in last 30 days)
                                    not_before = cert.get('notBefore')
                                    if not_before:
                                        issue_date = datetime.strptime(not_before, '%b %d %H:%M:%S %Y %Z')
                                        days_since_issue = (datetime.now() - issue_date).days
                                        if days_since_issue < 30:
                                            new_certificate = True
                                    
                                except Exception as e:
                                    self.logger.debug(f"Error parsing certificate dates: {e}")
                            
                            # Check issuer
                            issuer = cert.get('issuer')
                            if issuer:
                                issuer_str = str(issuer)
                                # Check for suspicious issuers (self-signed, unknown CAs)
                                if 'self' in issuer_str.lower() or 'localhost' in issuer_str.lower():
                                    suspicious_certificate = True
                                    certificate_issues.append("Self-signed certificate")
                            
                            # Check subject
                            subject = cert.get('subject')
                            if subject:
                                subject_str = str(subject)
                                # Check if certificate subject doesn't match domain
                                if domain not in subject_str.lower():
                                    suspicious_certificate = True
                                    certificate_issues.append("Certificate subject mismatch")
                        
            except ssl.SSLError as e:
                suspicious_certificate = True
                certificate_issues.append(f"SSL error: {str(e)}")
            except socket.timeout:
                certificate_issues.append("Connection timeout")
            except Exception as e:
                self.logger.debug(f"Certificate check error for {domain}: {e}")
                certificate_issues.append(f"Certificate check failed: {str(e)}")
            
            # Also try HTTP request to verify HTTPS works
            try:
                response = await self.client.head(url, timeout=5)
                if response.status_code >= 400:
                    certificate_issues.append(f"HTTPS request failed with status {response.status_code}")
            except httpx.RequestError:
                pass  # Already checked certificate directly
                
        except Exception as e:
            self.logger.debug(f"HTTPS/certificate check error for {domain}: {e}")
            certificate_issues.append(f"Check error: {str(e)}")

        return {
            "uses_https": uses_https,
            "suspicious_certificate": suspicious_certificate,
            "new_certificate": new_certificate,
            "certificate_issues": certificate_issues if certificate_issues else []
        }

    async def _check_domain_age_and_reputation(self, domain: str) -> Dict[str, Any]:
        """
        Check domain age and reputation signals.
        
        Uses DNS lookups and heuristics to estimate domain age and reputation.
        In production, would integrate with WHOIS APIs and threat intelligence feeds.
        """
        new_domain_risk = False
        suspicious_reputation = False
        estimated_age_days = None
        
        # Check cache first
        if domain in self.domain_age_cache:
            cached_data = self.domain_age_cache[domain]
            return {
                "new_domain_risk": cached_data.get("new_domain_risk", False),
                "suspicious_reputation": cached_data.get("suspicious_reputation", False),
                "estimated_age_days": cached_data.get("estimated_age_days")
            }
        
        try:
            # Try DNS lookup to check if domain exists
            try:
                socket.gethostbyname(domain)
            except socket.gaierror:
                # Domain doesn't resolve - suspicious
                suspicious_reputation = True
                new_domain_risk = True
                estimated_age_days = 0
            else:
                # Domain resolves - estimate age based on heuristics
                
                # Heuristic 1: Suspicious keywords in domain name
                suspicious_keywords = [
                    "new", "free", "offer", "rewards", "login", "verify", "secure",
                    "update", "claim", "win", "prize", "lottery", "urgent",
                    "account", "bank", "payment", "refund", "suspended"
                ]
                
                domain_lower = domain.lower()
                if any(keyword in domain_lower for keyword in suspicious_keywords):
                    new_domain_risk = True
                    estimated_age_days = 30  # Estimate as new
                
                # Heuristic 2: Suspicious TLD
                tld = domain.split('.')[-1] if '.' in domain else ""
                if tld in SUSPICIOUS_TLDS:
                    new_domain_risk = True
                    suspicious_reputation = True
                    if estimated_age_days is None:
                        estimated_age_days = 60
                
                # Heuristic 3: Domain length and structure
                domain_name = domain.split('.')[0] if '.' in domain else domain
                if len(domain_name) < 4 or len(domain_name) > 30:
                    suspicious_reputation = True
                
                # Heuristic 4: Random character patterns (often used in new phishing domains)
                if re.search(r'[a-z]{1,3}[0-9]{3,}', domain_name):
                    new_domain_risk = True
                    if estimated_age_days is None:
                        estimated_age_days = 45
                
                # Default: if no indicators, assume older domain
                if estimated_age_days is None:
                    estimated_age_days = 365  # Assume 1 year old
                
                # New domain risk if estimated age < 90 days
                if estimated_age_days < 90:
                    new_domain_risk = True
                
        except Exception as e:
            self.logger.debug(f"Domain age/reputation check error for {domain}: {e}")
            # Default to suspicious if check fails
            new_domain_risk = True
            estimated_age_days = 30
        
        # Cache result
        result = {
            "new_domain_risk": new_domain_risk,
            "suspicious_reputation": suspicious_reputation,
            "estimated_age_days": estimated_age_days
        }
        self.domain_age_cache[domain] = result
        
        return result

    async def _analyze_redirections(self, url: str) -> Dict[str, Any]:
        """
        Safe redirection analysis using HEAD requests only.
        
        Detects:
        - Excessive redirects (suspicious chains)
        - Mismatched final domains
        - Redirect chains
        - Final destination analysis
        """
        redirect_count = 0
        excessive_redirects = False
        mismatched_final_domain = False
        redirect_chain_detected = False
        final_url = url
        redirect_chain = []
        initial_domain = urlparse(url).netloc.lower()
        
        try:
            # Follow redirects manually to analyze chain
            current_url = url
            max_redirects = 10  # Safety limit
            visited_urls = set()
            
            for i in range(max_redirects):
                if current_url in visited_urls:
                    redirect_chain_detected = True
                    break
                
                visited_urls.add(current_url)
                
                try:
                    # Use HEAD request (safe, read-only)
                    response = await self.client.head(current_url, follow_redirects=False, timeout=5)
                    
                    # Check for redirect status codes
                    if response.status_code in [301, 302, 303, 307, 308]:
                        redirect_count += 1
                        redirect_chain_detected = True
                        
                        # Get redirect location
                        location = response.headers.get('Location') or response.headers.get('location')
                        if location:
                            # Handle relative URLs
                            if location.startswith('/'):
                                parsed = urlparse(current_url)
                                location = f"{parsed.scheme}://{parsed.netloc}{location}"
                            elif not location.startswith(('http://', 'https://')):
                                parsed = urlparse(current_url)
                                location = f"{parsed.scheme}://{parsed.netloc}/{location}"
                            
                            redirect_chain.append({
                                "from": current_url,
                                "to": location,
                                "status": response.status_code
                            })
                            
                            current_url = location
                            continue
                    
                    # No redirect, this is the final URL
                    final_url = str(response.url) if hasattr(response, 'url') else current_url
                    break
                    
                except httpx.HTTPStatusError:
                    # Non-redirect status, stop following
                    final_url = current_url
                    break
                except httpx.RequestError as e:
                    self.logger.debug(f"Redirect check error at step {i}: {e}")
                    final_url = current_url
                    break
            
            # Analyze redirect chain
            if redirect_count > 3:
                excessive_redirects = True
            
            # Check final domain
            final_parsed = urlparse(final_url)
            final_domain = final_parsed.netloc.lower() if final_parsed.netloc else ""
            
            if final_domain and initial_domain != final_domain:
                # Allow legitimate shorteners
                if final_domain not in SHORTLINK_DOMAINS and initial_domain not in SHORTLINK_DOMAINS:
                mismatched_final_domain = True

        except Exception as e:
            self.logger.debug(f"Redirection analysis error for {url}: {e}")

        return {
            "redirect_count": redirect_count,
            "excessive_redirects": excessive_redirects,
            "mismatched_final_domain": mismatched_final_domain,
            "redirect_chain_detected": redirect_chain_detected,
            "final_url": final_url,
            "final_domain": final_domain if 'final_domain' in locals() else None,
            "redirect_chain": redirect_chain
        }

    async def _fetch_content_metadata(self, url: str) -> Dict[str, Any]:
        """
        Safe content semantic fingerprinting.
        
        Fetches minimal page metadata (title, headers, forms) to detect:
        - Login/credential forms
        - Payment gateways
        - Brand impersonation
        - Suspicious content patterns
        
        NO JavaScript execution, NO form submission, NO user tracking.
        """
        login_form_detected = False
        brand_impersonation_detected = False
        payment_gateway_detected = False
        page_title = ""
        impersonated_brand = None
        content_indicators = []
        
        try:
            # Fetch only first 10KB for metadata analysis (safe, read-only)
            max_bytes = 10240
            content = b""
            
            async with self.client.stream("GET", url, timeout=5, follow_redirects=False) as response:
                if response.status_code >= 400:
                    return {
                        "login_form_detected": False,
                        "brand_impersonation_detected": False,
                        "payment_gateway_detected": False,
                        "page_title": "",
                        "impersonated_brand": None
                    }
                
                async for chunk in response.aiter_bytes():
                    content += chunk
                    if len(content) >= max_bytes:
                        break
            
            # Decode content (handle encoding errors gracefully)
            try:
                decoded_content = content.decode('utf-8', errors='ignore').lower()
            except:
                decoded_content = content.decode('latin-1', errors='ignore').lower()
            
            # 1. DETECT LOGIN/CREDENTIAL FORMS
            login_indicators = [
                r'<input[^>]*type=["\']?password["\']?[^>]*>',
                r'<input[^>]*name=["\']?(?:password|pwd|pass|passwd)["\']?[^>]*>',
                r'<form[^>]*(?:login|signin|authenticate|verify)[^>]*>',
                r'login\s+form|sign\s+in|enter\s+password|password\s+field',
                r'<input[^>]*type=["\']?email["\']?[^>]*>.*?<input[^>]*type=["\']?password["\']?[^>]*>',
            ]
            
            for pattern in login_indicators:
                if re.search(pattern, decoded_content, re.IGNORECASE | re.DOTALL):
                        login_form_detected = True
                    content_indicators.append("login_form")
                    break
            
            # 2. DETECT PAYMENT GATEWAYS
            payment_indicators = [
                r'<input[^>]*type=["\']?text["\']?[^>]*name=["\']?(?:card|credit|debit|cvv|cvc)["\']?[^>]*>',
                r'payment\s+form|credit\s+card|debit\s+card|card\s+number',
                r'<input[^>]*name=["\']?(?:cardnumber|card_number|ccnumber)["\']?[^>]*>',
                r'expiry\s+date|expiration|mm/yy|mm/yyyy',
            ]
            
            for pattern in payment_indicators:
                if re.search(pattern, decoded_content, re.IGNORECASE):
                    payment_gateway_detected = True
                    content_indicators.append("payment_form")
                    break
            
            # 3. DETECT BRAND IMPERSONATION
                        parsed_url = urlparse(url)
            url_domain = parsed_url.netloc.lower()
            url_domain_parts = url_domain.split('.')
            url_base = url_domain_parts[-2] if len(url_domain_parts) >= 2 else url_domain
            
            # Check if page content mentions known brands but domain doesn't match
            for brand_domain in self.known_legitimate_domains:
                brand_name = brand_domain.split('.')[0]
                
                # Check if brand name appears in content
                if brand_name in decoded_content:
                    # Check if domain doesn't match brand
                    if brand_name not in url_base and url_base not in brand_domain:
                        # Additional check: look for brand-specific keywords
                        brand_keywords = {
                            "google": ["gmail", "google account", "google sign in"],
                            "microsoft": ["microsoft account", "outlook", "office 365"],
                            "amazon": ["amazon account", "amazon sign in"],
                            "paypal": ["paypal account", "paypal login"],
                            "facebook": ["facebook login", "facebook account"],
                        }
                        
                        if brand_name in brand_keywords:
                            keywords = brand_keywords[brand_name]
                            if any(kw in decoded_content for kw in keywords):
                                brand_impersonation_detected = True
                                impersonated_brand = brand_name
                                content_indicators.append(f"brand_impersonation_{brand_name}")
                                break
                        else:
                            # Generic brand mention check
                            if f"{brand_name} login" in decoded_content or f"{brand_name} sign in" in decoded_content:
                            brand_impersonation_detected = True
                                impersonated_brand = brand_name
                                content_indicators.append(f"brand_impersonation_{brand_name}")
                                break
            
            # 4. EXTRACT PAGE TITLE
            title_match = re.search(r'<title[^>]*>(.*?)</title>', decoded_content, re.IGNORECASE | re.DOTALL)
            if title_match:
                page_title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip()[:200]  # Limit length
            
            # 5. CHECK FOR SUSPICIOUS CONTENT PATTERNS
            suspicious_patterns = [
                r'urgent.*action.*required',
                r'your\s+account\s+has\s+been\s+suspended',
                r'verify\s+your\s+identity',
                r'click\s+here\s+to\s+verify',
                r'limited\s+time\s+offer',
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, decoded_content, re.IGNORECASE):
                    content_indicators.append("suspicious_content")
                        break

        except httpx.RequestError as e:
            self.logger.debug(f"Content metadata fetch failed for {url}: {e}")
        except Exception as e:
            self.logger.error(f"Error fetching content metadata for {url}: {e}")

        return {
            "login_form_detected": login_form_detected,
            "brand_impersonation_detected": brand_impersonation_detected,
            "payment_gateway_detected": payment_gateway_detected,
            "page_title": page_title,
            "impersonated_brand": impersonated_brand,
            "content_indicators": content_indicators
        }

    def extract_urls(self, text: str) -> List[str]:
        """
        Extract all URLs, domain mentions, and QR code references from text.

        Detects:
        - Full URLs (http://, https://)
        - Domain mentions (e.g., "visit example.com")
        - Direct domains (e.g., "example.com")
        - Shortened links
        - QR code references (textual)
        """
        urls = []

        try:
            # 1. Extract full URLs (http://, https://)
            full_urls = URL_PATTERN.findall(text)
            urls.extend(full_urls)

            # 2. Extract domain mentions with action verbs (e.g., "visit bank-site.com")
            domain_mentions = DOMAIN_MENTION_PATTERN.findall(text)
            for domain in domain_mentions:
                # Normalize to full URL
                if domain and '.' in domain:
                    # Skip if already found as full URL
                    if not any(domain in url for url in full_urls):
                        urls.append(f"https://{domain}")

            # 3. Extract suspicious direct domain mentions (e.g., "bank-secure.net")
            # Only add domains that look suspicious (contain hyphens, numbers, or are subdomains)
            direct_domains = DIRECT_DOMAIN_PATTERN.findall(text)
            for domain in direct_domains:
                if (domain and '.' in domain and
                    len(domain) > 4 and
                    not any(domain in url for url in urls) and
                    ('-' in domain or len(domain.split('.')) > 2 or any(c.isdigit() for c in domain))):
                    urls.append(f"https://{domain}")

            # 4. Special handling for known shorteners without http:// prefix
            shortener_domains = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly']
            for shortener in shortener_domains:
                if shortener in text.lower():
                    # Look for patterns like "bit.ly/xyz123"
                    import re as re_module
                    pattern = rf'\b{re_module.escape(shortener)}/[^\s]+'
                    matches = re_module.findall(pattern, text.lower())
                    for match in matches:
                        if not any(match in url for url in urls):
                            urls.append(f"https://{match}")

        except Exception as e:
            # Fallback: basic URL detection
            self.logger.warning(f"URL extraction failed, using fallback: {e}")
            basic_urls = URL_PATTERN.findall(text)
            urls.extend(basic_urls)

        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)

        return unique_urls
    
    def extract_domain_mentions(self, text: str) -> List[str]:
        """Extract domain mentions that request user action."""
        mentions = []
        matches = DOMAIN_MENTION_PATTERN.findall(text)
        for domain in matches:
            if domain and '.' in domain:
                mentions.append(domain)
        return mentions
    
    def has_qr_reference(self, text: str) -> bool:
        """Check if text contains QR code references."""
        return bool(QR_REFERENCE_PATTERN.search(text))

    def _calculate_levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculates the Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._calculate_levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

url_analyzer = URLAnalyzer()
