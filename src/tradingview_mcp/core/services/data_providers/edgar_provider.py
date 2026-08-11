"""
SEC EDGAR — free, official, full XBRL company filings. No paid tier, no API
key — replaces FMP/Intrinio premium fundamentals per the OpenBB
free-substitution plan.

SEC's fair-use policy requires a descriptive User-Agent identifying the
requester (see https://www.sec.gov/os/webmaster-faq#developers) — set
SEC_EDGAR_USER_AGENT to "Your Name your@email.com" or requests may be
throttled/blocked. Refuses to call out with the generic default.

Two endpoints used:
  - company_tickers.json: free ticker -> CIK lookup table (cached in-process).
  - companyconcept API: one XBRL us-gaap concept's full history for a company
    (e.g. Revenues, Assets, NetIncomeLoss) — chosen over the full
    companyfacts dump, which can be multiple MB per company.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Optional

from tradingview_mcp.core.errors import ErrorCode, make_error

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"

_ticker_cik_cache: Optional[dict[str, int]] = None


def _user_agent() -> Optional[str]:
    ua = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
    return ua or None


def _load_ticker_cik_map(user_agent: str) -> dict[str, int]:
    global _ticker_cik_cache
    if _ticker_cik_cache is not None:
        return _ticker_cik_cache
    req = urllib.request.Request(_TICKERS_URL, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    _ticker_cik_cache = {row["ticker"].upper(): row["cik_str"] for row in data.values()}
    return _ticker_cik_cache


def get_company_fundamentals(
    ticker: str,
    concept: str = "Revenues",
    user_agent: Optional[str] = None,
) -> dict[str, Any]:
    """Fetch one XBRL us-gaap concept's reported history for a US-listed
    company from SEC EDGAR (e.g. concept='Assets', 'NetIncomeLoss',
    'EarningsPerShareBasic', 'Revenues', 'StockholdersEquity').

    Args:
        ticker: stock ticker, e.g. 'AAPL'.
        concept: an XBRL us-gaap taxonomy tag — see
            https://www.sec.gov/cgi-bin/browse-edgar for a company's filings
            to find the tags it actually reports.
        user_agent: overrides SEC_EDGAR_USER_AGENT env var for this call.
    """
    ua = user_agent or _user_agent()
    if not ua:
        return make_error(
            ErrorCode.DEPENDENCY_MISSING,
            "SEC_EDGAR_USER_AGENT is not set. SEC requires a descriptive "
            "User-Agent (e.g. 'Your Name your@email.com') for API access — "
            "see https://www.sec.gov/os/webmaster-faq#developers",
        )
    if not ticker:
        return make_error(ErrorCode.INVALID_PARAMETER, "ticker is required")

    try:
        cik_map = _load_ticker_cik_map(ua)
    except Exception as e:
        return make_error(ErrorCode.UPSTREAM_ERROR, f"failed to load SEC ticker->CIK map: {e}", retryable=True)

    cik = cik_map.get(ticker.upper())
    if cik is None:
        return make_error(ErrorCode.SYMBOL_NOT_FOUND, f"'{ticker}' not found in SEC EDGAR's ticker list")

    url = _CONCEPT_URL.format(cik=str(cik).zfill(10), concept=concept)
    req = urllib.request.Request(url, headers={"User-Agent": ua})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return make_error(
                ErrorCode.NO_DATA,
                f"'{ticker}' has no reported XBRL concept '{concept}' (company may use a "
                f"different tag, or may not be a US-GAAP filer)",
            )
        body = e.read().decode("utf-8", errors="replace")
        return make_error(ErrorCode.UPSTREAM_ERROR, f"SEC EDGAR API error {e.code}: {body[:300]}")
    except Exception as e:
        return make_error(ErrorCode.UPSTREAM_ERROR, f"SEC EDGAR request failed: {e}", retryable=True)

    units = data.get("units", {})
    facts = []
    for unit_name, entries in units.items():
        for e in entries:
            facts.append({
                "unit": unit_name,
                "fiscal_year": e.get("fy"),
                "fiscal_period": e.get("fp"),
                "start": e.get("start"),
                "end": e.get("end"),
                "value": e.get("val"),
                "form": e.get("form"),
                "filed": e.get("filed"),
            })
    facts.sort(key=lambda f: f["end"] or "")

    return {
        "ticker": ticker.upper(),
        "cik": str(cik).zfill(10),
        "concept": concept,
        "label": data.get("label"),
        "description": data.get("description"),
        "n_facts": len(facts),
        "facts": facts,
        "source": "SEC EDGAR (data.sec.gov)",
    }
