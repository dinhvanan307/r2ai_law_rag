"""Polite crawling + HTML import utilities for building the legal corpus.

Design principles
-----------------
* **Respectful**: honors robots.txt, rate-limits, identifies itself, retries
  with backoff, and caches raw HTML so re-runs never re-hit the server.
* **No access-control bypass**: never logs in, never solves CAPTCHAs, never
  touches paywalled content. For login-only sources, use ``html_import`` on
  pages the user saved themselves via their own authenticated session.
* **Robust to DOM drift**: metadata (law_id / doc_type / title) comes from the
  curated seed list, so we only need the document *full text* from the page.
"""
