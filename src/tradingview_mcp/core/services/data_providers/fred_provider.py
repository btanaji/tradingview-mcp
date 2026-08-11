"""
FRED (Federal Reserve Economic Data, St. Louis Fed) — macro/economic series.

Free, official, no paid tier — the OpenBB architecture doc's "no substitute
needed" case. Needs a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
(instant signup, no cost) set as FRED_API_KEY.

Uses urllib directly (no `fredapi` dependency) — one JSON GET, consistent
with how the rest of this codebase talks to free HTTP APIs
(marketaux_service.py, backtest_service._fetch_ohlcv).
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any, Optional

from tradingview_mcp.core.errors import ErrorCode, make_error

_UA = "tradingview-mcp/1.0 fred-client"
_BASE = "https://api.stlouisfed.org/fred/series/observations"


def get_fred_series(
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
    limit: int = 1000,
) -> dict[str, Any]:
    """Fetch observations for a FRED series (e.g. 'GDP', 'CPIAUCSL', 'DFF'
    for the fed funds rate, 'UNRATE' for unemployment).

    Args:
        series_id: FRED series ID — browse them at https://fred.stlouisfed.org/.
        start_date / end_date: 'YYYY-MM-DD', both optional (defaults to full history).
        api_key: overrides the FRED_API_KEY env var for this call.
        limit: max observations to return (FRED default page size is 100000; capped
            here at 1000 to keep tool responses small — use start_date/end_date to scope).
    """
    key = api_key or os.environ.get("FRED_API_KEY", "")
    if not key:
        return make_error(
            ErrorCode.DEPENDENCY_MISSING,
            "FRED_API_KEY is not set. Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html",
        )
    if not series_id:
        return make_error(ErrorCode.INVALID_PARAMETER, "series_id is required")

    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "limit": limit,
        "sort_order": "asc",
    }
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date

    url = f"{_BASE}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return make_error(ErrorCode.UPSTREAM_ERROR, f"FRED API error {e.code}: {body[:300]}")
    except Exception as e:
        return make_error(ErrorCode.UPSTREAM_ERROR, f"FRED request failed: {e}", retryable=True)

    observations = data.get("observations", [])
    series = [
        {"date": o["date"], "value": float(o["value"])}
        for o in observations
        if o.get("value") not in (None, ".")
    ]
    return {
        "series_id": series_id,
        "n_observations": len(series),
        "observations": series,
        "source": "FRED (St. Louis Fed)",
    }
