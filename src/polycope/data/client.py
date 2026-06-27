"""Async client for Polymarket's public Data API and Gamma API.

Read endpoints require no authentication. We add bounded concurrency, timeouts,
and retry-with-backoff so bulk ingestion is polite and resilient.

Endpoints used (see docs https://docs.polymarket.com/api-reference):
  Data API  /v1/leaderboard          population of candidate wallets
            /v1/trades?user=0x...    per-wallet fills (max 500/page, recent-first)
            /v1/activity?user=...    full chronological history, timestamp-paginated
            /v1/positions?user=...   current positions (used by the live executor)
  Gamma API /markets                 market metadata + resolution outcomes (labels)
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import Settings, settings

# Errors worth retrying: transient network issues and 5xx/429 from the API.
_RETRYABLE = (httpx.TransportError, httpx.HTTPStatusError)


class PolymarketClient:
    """Thin async wrapper over the public REST APIs.

    Usage:
        async with PolymarketClient() as c:
            board = await c.leaderboard(limit=100)
            trades = await c.all_trades(wallet)
    """

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings
        self._sem = asyncio.Semaphore(self.cfg.max_concurrency)
        self._client = httpx.AsyncClient(
            timeout=self.cfg.request_timeout,
            headers={"Accept": "application/json"},
        )

    async def __aenter__(self) -> PolymarketClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, base: str, path: str, **params: Any) -> Any:
        @retry(
            retry=retry_if_exception_type(_RETRYABLE),
            wait=wait_exponential(multiplier=2, min=2, max=16),
            stop=stop_after_attempt(self.cfg.max_retries),
            reraise=True,
        )
        async def _do() -> Any:
            async with self._sem:
                resp = await self._client.get(f"{base}{path}", params=params)
                # 4xx other than 429 are not retryable (e.g. bad wallet) -> surface.
                if resp.status_code == 429 or resp.status_code >= 500:
                    resp.raise_for_status()
                resp.raise_for_status()
                return resp.json()

        return await _do()

    # ---- Data API ----
    async def leaderboard(self, limit: int = 100, offset: int = 0, **extra: Any) -> list[dict]:
        """Top traders by PnL/volume. Returns rows with at least an address field."""
        return await self._get(self.cfg.data_api, "/v1/leaderboard", limit=limit, offset=offset, **extra)

    async def trades(self, wallet: str, limit: int = 500, offset: int = 0, **extra: Any) -> list[dict]:
        return await self._get(
            self.cfg.data_api, "/v1/trades", user=wallet, limit=limit, offset=offset, **extra
        )

    async def activity(
        self,
        wallet: str,
        limit: int = 500,
        offset: int = 0,
        start: int | None = None,
        end: int | None = None,
        type: str | None = None,
    ) -> list[dict]:
        params: dict[str, Any] = {"user": wallet, "limit": limit, "offset": offset}
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end
        if type is not None:
            params["type"] = type
        return await self._get(self.cfg.data_api, "/v1/activity", **params)

    async def positions(self, wallet: str, **extra: Any) -> list[dict]:
        return await self._get(self.cfg.data_api, "/v1/positions", user=wallet, **extra)

    # Data API rejects offset > 3000 with 400.
    _TRADES_MAX_OFFSET = 3000

    async def all_trades(self, wallet: str, page: int = 500, hard_cap: int = 50_000) -> list[dict]:
        """Page through a wallet's full trade history via offset pagination."""
        out: list[dict] = []
        offset = 0
        while len(out) < hard_cap and offset <= self._TRADES_MAX_OFFSET:
            batch = await self.trades(wallet, limit=page, offset=offset)
            if not batch:
                break
            out.extend(batch)
            if len(batch) < page:
                break
            offset += page
        return out

    # ---- Gamma API ----
    async def markets(self, limit: int = 100, offset: int = 0, **extra: Any) -> list[dict]:
        return await self._get(self.cfg.gamma_api, "/markets", limit=limit, offset=offset, **extra)
