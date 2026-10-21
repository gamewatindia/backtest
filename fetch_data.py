#!/usr/bin/env python3
"""
Download OHLCV history from MEXC or Gate.io public APIs (no API key needed)
and save as data/SYMBOL_exchange.csv for the backtest scripts.

  python fetch_data.py --exchange mexc --symbols BTC_USDT,ETH_USDT --days 365
  python fetch_data.py --exchange gate --symbols BTC_USDT --days 30
  python fetch_data.py --exchange gate --symbols BTC_USDT --days 365 --interval 1h

Speed: coins are downloaded in parallel (--workers, default 5) and failed requests
are retried automatically. 5m x 365 days is ~1.05 lakh candles per coin (~210
requests); with 5 workers this takes a couple of minutes instead of 5-10.

IMPORTANT (Gate): Gate only serves the last ~10,000 candles per interval, so
5m data reaches back only ~34 days. For longer history use --interval 1h
(~1 year) or 4h/1d, or use MEXC.
"""
import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

SEC = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
MEXC_IV = {"5m": "5m", "15m": "15m", "1h": "60m", "4h": "4h", "1d": "1d"}
GATE_MAX_BACK = 9800  # Gate serves at most 10,000 candles back
_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = "brahmastra-backtest/1.2"
    return _local.s


def get_json(url, params, tries=6):
    """GET with retries on rate limits / server errors / network hiccups."""
    last = None
    for attempt in range(tries):
        try:
            r = session().get(url, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code not in (418, 429, 500, 502, 503, 504):
                break                                   # real error (bad pair etc.) - do not retry
        except requests.RequestException as e:
            last = f"network error: {e}"
        time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(last)


def mexc(sym, start_ms, end_ms, iv):
    url = "https://api.mexc.com/api/v3/klines"
    pair = sym.replace("_", "")
    step = SEC[iv] * 1000
    rows, t = [], start_ms
    # Explicit window (startTime AND endTime) per request: with only startTime some
    # servers return just the latest 500 candles.
    while t < end_ms:
        te = min(t + 500 * step - 1, end_ms)
        k = get_json(url, dict(symbol=pair, interval=MEXC_IV[iv], startTime=t, endTime=te, limit=500))
        rows += [(x[0], x[1], x[2], x[3], x[4], x[5]) for x in k if t <= x[0] <= te]
        t = te + 1
        time.sleep(0.05)
    return rows


def gate(sym, start_ms, end_ms, iv):
    url = "https://api.gateio.ws/api/v4/spot/candlesticks"
    sec = SEC[iv]
    earliest = end_ms - GATE_MAX_BACK * sec * 1000
    if start_ms < earliest:
        days = GATE_MAX_BACK * sec / 86400
        print(f"  note: Gate keeps only ~{days:.0f} days of {iv} candles -> downloading that much "
              f"(use --interval 1h/4h/1d or MEXC for longer history)", flush=True)
        start_ms = earliest
    rows, t = [], start_ms // 1000
    end_s = end_ms // 1000
    while t < end_s:
        to = min(t + sec * 900, end_s)  # < 1000 candles per request
        data = get_json(url, {"currency_pair": sym, "interval": iv, "from": t, "to": to})
        # Gate row: [ts, quote_vol, close, high, low, open, base_vol, window_closed]
        for x in data:
            if len(x) > 7 and str(x[7]).lower() == "false":
                continue  # skip the still-forming candle
            rows.append((int(x[0]) * 1000, x[5], x[3], x[4], x[2], x[6]))
        t = to + 1
        time.sleep(0.05)
    return rows


def fetch_one(fn, sym, start, end, iv):
    t0 = time.time()
    rows = fn(sym, start, end, iv)
    return sym, rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML settings file (see config.yml)")
    ap.add_argument("--exchange", choices=["mexc", "gate"])
    ap.add_argument("--symbols", help="comma list, e.g. BTC_USDT,ETH_USDT")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--interval", choices=list(SEC), default="5m")
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=5, help="coins downloaded in parallel")
    a = ap.parse_args()
    if a.config:
        import yaml
        with open(a.config) as fh:
            d = (yaml.safe_load(fh) or {}).get("data", {})
        a.exchange = a.exchange or d.get("exchange")
        a.symbols = a.symbols or ",".join(d.get("symbols", []))
        a.days = d.get("days", a.days) if a.days == 365 else a.days
        a.out = d.get("folder", a.out) if a.out == "data" else a.out
        if a.interval == "5m" and d.get("interval") in SEC:
            a.interval = d["interval"]
    if not a.exchange or not a.symbols:
        ap.error("need --exchange and --symbols (or --config)")

    os.makedirs(a.out, exist_ok=True)
    end = int(time.time() * 1000)
    start = end - a.days * 86400 * 1000
    fn = mexc if a.exchange == "mexc" else gate
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    est = a.days * 86400 / SEC[a.interval]
    print(f"Downloading {len(syms)} coin(s) from {a.exchange}, {a.interval} candles, {a.days} days "
          f"(~{est:,.0f} candles per coin), {min(a.workers, len(syms))} in parallel ...", flush=True)

    ok = 0
    with ThreadPoolExecutor(max_workers=max(1, min(a.workers, len(syms)))) as ex:
        futs = {ex.submit(fetch_one, fn, s, start, end, a.interval): s for s in syms}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                _, rows, secs = fut.result()
            except Exception as e:
                print(f"{sym}: FAILED -> {e}", flush=True)
                continue
            if not rows:
                print(f"{sym}: FAILED -> no candles returned", flush=True)
                continue
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = df.drop_duplicates("timestamp").sort_values("timestamp")
            df.to_csv(os.path.join(a.out, f"{sym.replace('_', '')}_{a.exchange}.csv"), index=False)
            first = pd.to_datetime(df.timestamp.iloc[0], unit="ms")
            last = pd.to_datetime(df.timestamp.iloc[-1], unit="ms")
            got_days = (last - first).total_seconds() / 86400
            print(f"{sym}: {len(df)} candles saved ({a.interval}), {first:%Y-%m-%d} -> {last:%Y-%m-%d}  [{secs:.0f}s]",
                  flush=True)
            if got_days < a.days * 0.9 and not (a.exchange == "gate" and a.interval == "5m"):
                print(f"  WARNING: asked {a.days} days but only got {got_days:.0f} days - exchange returned less history",
                      flush=True)
            ok += 1
    if ok == 0:
        sys.exit("No data downloaded - see errors above.")


if __name__ == "__main__":
    main()
