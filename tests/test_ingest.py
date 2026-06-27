"""Normalizers must tolerate the API's field-name drift and coerce types."""

from polycope.data.ingest import extract_wallets, normalize_trade, trades_to_frame
from polycope.data.resolutions import normalize_market


def test_normalize_trade_variants():
    raw = {"proxyWallet": "0xABC", "conditionId": "0xMKT", "outcomeIndex": 1,
           "side": "BUY", "price": "0.42", "size": "100", "timestamp": 1700, "transactionHash": "0xT"}
    row = normalize_trade(raw)
    assert row["wallet"] == "0xabc"
    assert row["market_id"] == "0xMKT"
    assert row["outcome_index"] == 1
    assert row["side"] == "BUY"
    assert row["price"] == 0.42
    assert row["size"] == 100.0


def test_normalize_trade_missing_price_returns_none():
    assert normalize_trade({"size": 10, "side": "BUY"}) is None


def test_trades_to_frame_filters_bad_rows():
    raws = [
        {"wallet": "0x1", "market": "m", "side": "BUY", "price": 0.5, "size": 10, "timestamp": 1},
        {"wallet": "0x2", "market": "m", "side": "BUY", "price": 1.5, "size": 10, "timestamp": 1},  # price>1
        {"wallet": "0x3", "market": "m", "side": "SELL", "price": 0.5, "size": 0, "timestamp": 1},  # size 0
    ]
    df = trades_to_frame(raws)
    assert len(df) == 1
    assert df.iloc[0]["wallet"] == "0x1"


def test_extract_wallets_dedupes():
    rows = [{"proxyWallet": "0xA"}, {"address": "0xB"}, {"wallet": "0xa"}]
    assert extract_wallets(rows) == ["0xa", "0xb"]


def test_normalize_market_resolution_from_prices():
    raw = {"conditionId": "0xM", "question": "Will X?", "closed": True,
           "outcomePrices": "[\"1\", \"0\"]", "endDate": 1800, "startDate": 1000}
    m = normalize_market(raw)
    assert m["resolved"] is True
    assert m["winning_outcome"] == 0
    assert m["market_id"] == "0xM"
