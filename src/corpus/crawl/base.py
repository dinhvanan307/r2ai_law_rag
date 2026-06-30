"""Polite HTTP session: rate limiting, retries, on-disk cache, robots.txt.

This keeps crawling respectful and reproducible. Raw HTML is cached by URL so
parser changes don't require re-fetching, and the target server is hit at most
once per document.
"""
from __future__ import annotations

import hashlib
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests

DEFAULT_UA = (
    "R2AI-LegalCorpusBot/0.1 (+research; contact: your-email@example.com) "
    "python-requests"
)


class PoliteSession:
    """A thin wrapper around requests with crawl etiquette built in."""

    def __init__(
        self,
        cache_dir: str | Path = "data/cache_html",
        min_interval: float = 2.0,      # seconds between requests (be gentle)
        timeout: int = 30,
        max_retries: int = 3,
        backoff: float = 2.0,
        user_agent: str = DEFAULT_UA,
        respect_robots: bool = True,
        verify_ssl: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.respect_robots = respect_robots
        self.verify_ssl = verify_ssl
        self._last_request = 0.0
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent,
                                     "Accept-Language": "vi,en;q=0.8"})
        if not verify_ssl:
            # The target gov site (vbpl.vn) sometimes serves an expired cert.
            # Opt-in only; warn once and silence noisy per-request warnings.
            print("[crawl] WARNING: SSL verification DISABLED (--insecure). "
                  "Only use for trusted public sources like vbpl.vn.")
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:
                pass

    # --- robots ------------------------------------------------------------
    def _allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._robots.get(base)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                rp.read()
            except Exception:
                # If robots.txt is unreachable, default to allowed but cautious.
                rp = None  # type: ignore
            self._robots[base] = rp  # type: ignore
        if rp is None:
            return True
        return rp.can_fetch(self.session.headers["User-Agent"], url)

    # --- cache -------------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"{h}.html"

    # --- fetch -------------------------------------------------------------
    def get(self, url: str, use_cache: bool = True) -> str | None:
        """Return page HTML (cached if available). ``None`` if blocked/failed."""
        cache = self._cache_path(url)
        if use_cache and cache.exists():
            return cache.read_text(encoding="utf-8", errors="ignore")

        if not self._allowed(url):
            print(f"[crawl] robots.txt disallows: {url}")
            return None

        # Rate limit.
        elapsed = time.time() - self._last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout,
                                        verify=self.verify_ssl)
                self._last_request = time.time()
                if resp.status_code == 200:
                    html = resp.text
                    cache.write_text(html, encoding="utf-8")
                    return html
                last_err = f"HTTP {resp.status_code}"
                if resp.status_code in (403, 401):
                    print(f"[crawl] {last_err} (auth/forbidden) for {url} — "
                          f"this source may require login; use html_import.")
                    return None
            except requests.exceptions.SSLError as e:
                print(f"[crawl] SSL error for {url}: {e}\n"
                      f"        vbpl.vn's certificate may be expired. "
                      f"Re-run with --insecure to bypass for this public source.")
                return None
            except Exception as e:  # network error
                last_err = str(e)
            sleep = self.backoff ** attempt
            print(f"[crawl] attempt {attempt}/{self.max_retries} failed "
                  f"({last_err}); retry in {sleep:.0f}s")
            time.sleep(sleep)
        print(f"[crawl] giving up on {url}: {last_err}")
        return None
