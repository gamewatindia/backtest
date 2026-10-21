#!/usr/bin/env python3
"""
RSI DIVERGENCE STRATEGY - Backtest (futures style: fixed margin + leverage)
===========================================================================
Signals (every timeframe separately)
  BEARISH divergence -> SHORT : price makes a HIGHER high while RSI makes a LOWER high
  BULLISH divergence -> LONG  : price makes a LOWER low   while RSI makes a HIGHER low
  Highs/lows are swing pivots: the extreme of `pivot` candles on each side. A pivot
  is only known `pivot` candles after it happened, so the signal is confirmed at the
  close of that candle and the trade enters at the OPEN of the next one (no look-ahead).

Exits (each target 1%..10% is tested as its own scenario)
  TP        : fixed take-profit, N % price move from entry
  STOP LOSS : --stoploss on | off | both
     swing (default): just beyond the divergence extreme (the 2nd high/low) + 0.5 ATR
                      buffer; distance clamped to 0.5% .. 5% of price. If price breaks
                      the divergence extreme the idea is wrong, so that is the exit.
     atr   : --sl-atr x ATR      pct : fixed --sl-pct %
     OFF   : no stop loss; trade ends at TP, LIQUIDATION or after --max-hold candles.
  If SL and TP are both touched inside one candle, SL is assumed hit first.

Money model: fixed MARGIN per trade (default Rs 1000, no compounding).
  P&L(Rs) = margin x leverage x net price move (fees + slippage on the full position).
  Liquidation at ~ 1/leverage - maintenance margin against you = whole margin lost.

Validation: data is split by time (default last 30% = TEST). Results are also shown
  "target chosen on train, judged on test" so you can see if a setup survives data it
  was not picked on.

Data: CSV per symbol in a folder: timestamp,open,high,low,close,volume (UTC).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

TF_RULES = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D", "1w": "W-MON", "1M": "MS"}
TF_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080, "1M": 43200}
OHLCV = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
INF = 10 ** 9


# ----------------------------------------------------------------- data ----
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next(c for c in df.columns if c in ("timestamp", "time", "date", "datetime", "open_time"))
    t = df[tcol]
    if np.issubdtype(t.dtype, np.number):
        idx = pd.to_datetime(t, unit="ms" if t.iloc[0] > 1e11 else "s", utc=True)
    else:
        idx = pd.to_datetime(t, utc=True)
    cols = ["open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    df = df.set_index(idx.rename("ts"))[cols]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    df = df.astype(float).sort_index()
    return df[~df.index.duplicated()]


def resample(df, tf):
    return df.resample(TF_RULES[tf], label="left", closed="left").agg(OHLCV).dropna(subset=["open", "close"])


# ----------------------------------------------------------- indicators ----
def rsi_wilder(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.where(dn == 0, np.where(up > 0, 100.0, 50.0), rsi)
    return pd.Series(rsi, index=close.index).fillna(50.0).values


def atr_wilder(df, n=14):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().values


def pivots(high, low, left, right):
    """Swing high/low flags. Pivot at j needs `right` later candles -> confirmed at j+right."""
    hs, ls = pd.Series(high), pd.Series(low)
    lmax = hs.shift(1).rolling(left).max().values
    rmax = hs[::-1].shift(1).rolling(right).max()[::-1].values
    lmin = ls.shift(1).rolling(left).min().values
    rmin = ls[::-1].shift(1).rolling(right).min()[::-1].values
    ph = (high > lmax) & (high >= rmax)
    pl = (low < lmin) & (low <= rmin)
    return ph, pl


def find_signals(df, p):
    """Return list of divergence signals sorted by entry bar. Uses only data up to the confirmation bar."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    n = len(c)
    rsi = rsi_wilder(df["close"], p["rsi_len"])
    atr = atr_wilder(df, 14)
    ema = df["close"].ewm(span=p["ema_len"], adjust=False).mean().values
    vol = df["volume"].values
    vavg = pd.Series(vol).rolling(20).mean().values
    R = p["pivot"]
    ph, pl = pivots(h, l, R, R)
    warm = max(60, 3 * p["rsi_len"])
    if p["trend"] != "off":
        warm = max(warm, p["ema_len"])               # EMA needs time to settle
    sigs = []
    for side, flags in ((-1, ph), (1, pl)):          # -1: bearish at pivot highs -> short, +1: bullish -> long
        if side == -1 and not p["allow_short"] or side == 1 and not p["allow_long"]:
            continue
        idx = np.where(flags)[0]
        idx = idx[idx >= warm]
        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a
            if gap < p["min_gap"] or gap > p["max_gap"]:
                continue
            if side == -1:
                ok = (h[b] > h[a] * (1 + p["min_price_diff"] / 100)
                      and rsi[b] < rsi[a] - p["min_rsi_diff"] and rsi[a] >= p["rsi_ob"])
                extreme = h[b]
            else:
                ok = (l[b] < l[a] * (1 - p["min_price_diff"] / 100)
                      and rsi[b] > rsi[a] + p["min_rsi_diff"] and rsi[a] <= p["rsi_os"])
                extreme = l[b]
            if not ok:
                continue
            s_bar = b + R                              # bar at whose close the pivot is confirmed
            e = s_bar + 1                              # entry at next bar's open
            if e >= n:
                continue
            # ---- optional filters (all use data up to the signal bar only -> no look-ahead) ----
            if p["trend"] == "with":                   # trade only WITH the EMA trend (dip-buy / rally-sell)
                if (side == 1 and c[s_bar] <= ema[s_bar]) or (side == -1 and c[s_bar] >= ema[s_bar]):
                    continue
            elif p["trend"] == "against":              # only counter-trend (classic reversal) - for comparison
                if (side == 1 and c[s_bar] > ema[s_bar]) or (side == -1 and c[s_bar] < ema[s_bar]):
                    continue
            if p["vol_mult"] > 0 and not (vol[b] >= p["vol_mult"] * vavg[b]):
                continue                               # volume at the 2nd pivot must be >= vol_mult x 20-bar average
            sigs.append(dict(side=side, p1=a, p2=b, s=s_bar, e=e, extreme=extreme, atr=atr[s_bar]))
    sigs.sort(key=lambda x: x["e"])
    return sigs


# ------------------------------------------------------------ simulation ----
def simulate(sg, o, h, l, c, ts, p, tp_pct, lev, sl_on):
    n = len(c)
    s, e = sg["side"], sg["e"]
    slip, fee, tds = p["slip"] / 100, p["fee"] / 100, p["tds"] / 100
    margin, mmr = p["margin"], p["mmr"] / 100
    entry = o[e] * (1 + slip * s)

    sl_price, dist = None, np.nan
    if sl_on:
        if p["sl_type"] == "swing":
            sl0 = sg["extreme"] - s * p["sl_buffer"] * sg["atr"]
            dist = min(max(s * (entry - sl0) / entry, p["sl_min"] / 100), p["sl_max"] / 100)
        elif p["sl_type"] == "atr":
            dist = min(max(p["sl_atr"] * sg["atr"] / entry, p["sl_min"] / 100), p["sl_max"] / 100)
        else:
            dist = p["sl_pct"] / 100
        sl_price = entry * (1 - s * dist)
    liq = entry * (1 - s * max(1.0 / lev - mmr, 1e-9))
    tp = entry * (1 + s * tp_pct / 100)

    if sl_on:
        stop = max(sl_price, liq) if s == 1 else min(sl_price, liq)
        stop_is_sl = sl_price >= liq if s == 1 else sl_price <= liq
    else:
        stop, stop_is_sl = liq, False

    end = min(n - 1, e + p["max_hold"])
    hi, lo = h[e:end + 1], l[e:end + 1]
    if s == 1:
        sh, th = lo <= stop, hi >= tp
    else:
        sh, th = hi >= stop, lo <= tp
    ks, kt = int(sh.argmax()), int(th.argmax())
    ks = ks if sh[ks] else INF
    kt = kt if th[kt] else INF

    liquidated = False
    if ks <= kt and ks < INF:                          # stop first (also when same candle: conservative)
        k = e + ks
        gap = ks > 0
        if stop_is_sl:
            px = (min(o[k], stop) if s == 1 else max(o[k], stop)) if gap else stop
            reason = "SL"
        else:
            px, reason, liquidated = stop, "LIQUIDATION", True
    elif kt < INF:
        k = e + kt
        px = tp
        if kt > 0:
            px = max(o[k], tp) if s == 1 else min(o[k], tp)
        reason = "TP"
    else:
        k, px = end, c[end]
        reason = "TIMEOUT" if end == e + p["max_hold"] else "END"

    if liquidated:
        xpx, net, pnl = px, -1.0 / lev, -margin
    else:
        xpx = px * (1 - slip * s)
        net = s * (xpx / entry - 1) - 2 * fee - tds
        pnl = max(margin * lev * net, -margin)
    # worst unrealised loss while the trade was open (Rs, positive number, never more than the margin)
    if reason in ("SL", "LIQUIDATION"):
        mae = max(-pnl, 0.0)
    else:
        adv = (entry - l[e:k + 1].min()) / entry if s == 1 else (h[e:k + 1].max() - entry) / entry
        mae = min(margin * lev * max(adv, 0.0), margin)
    return dict(side="LONG" if s == 1 else "SHORT", entry_time=ts[e], exit_time=ts[k], entry=entry, exit=xpx,
                sl_price=sl_price if sl_on else np.nan, sl_pct=dist * 100 if sl_on else np.nan,
                tp_price=tp, exit_reason=reason, bars=k - e, price_net_pct=net * 100, pnl_inr=pnl,
                roi_margin_pct=pnl / margin * 100, mae_inr=mae, exit_idx=k)


def run_symbol(sym, base, tfs, p, scenarios, split_time):
    out, base_min = [], int(np.median(np.diff(base.index.values).astype("timedelta64[m]").astype(int)))
    for tf in tfs:
        if TF_MINUTES[tf] < base_min:
            continue
        df = resample(base, tf)
        if len(df) < 120:
            print(f"  {sym} {tf}: only {len(df)} candles - skipped (need >= 120)")
            continue
        o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
        sigs = find_signals(df, p)
        nl = sum(1 for x in sigs if x["side"] == 1)
        print(f"  {sym} {tf}: {len(df)} candles, {len(sigs)} divergence signals ({nl} long / {len(sigs) - nl} short)")
        ts = df.index
        for tp_pct, lev, sl_on in scenarios:
            last_exit = -1
            for sg in sigs:
                if sg["e"] <= last_exit:               # one position at a time per scenario
                    continue
                t = simulate(sg, o, h, l, c, ts, p, tp_pct, lev, sl_on)
                last_exit = t.pop("exit_idx")
                t.update(symbol=sym, tf=tf, tp_pct=tp_pct, lev=lev, stoploss="on" if sl_on else "off",
                         period="test" if t["entry_time"] >= split_time else "train")
                out.append(t)
    return out


# ------------------------------------------------------------- reporting ----
KEYS = ["tf", "stoploss", "lev", "tp_pct"]


def dd_stats(pnl, mae=None):
    """Max drawdown in Rs for a sequence of trades in time order.
    closed  : drawdown measured on closed-trade equity only
    floating: also counts the unrealised loss of each trade while it was open (matters most with stop loss OFF)"""
    pnl = np.asarray(pnl, float)
    cum = np.cumsum(pnl)
    c0 = np.r_[0.0, cum]
    pk = np.maximum.accumulate(c0)
    closed = float((pk[1:] - cum).max())
    if mae is None:
        return closed, closed
    floating = float((pk[:-1] - (c0[:-1] - np.asarray(mae, float))).max())
    return closed, max(closed, floating)


def drawdowns(tr):
    """-> DataFrame indexed by KEYS: max_dd_all (all coins combined, closed trades) and
    max_dd_coin (worst single coin, including unrealised loss of open trades)."""
    comb = {k: dd_stats(g.sort_values("exit_time").pnl_inr.values)[0] for k, g in tr.groupby(KEYS)}
    coin = {}
    for k, g in tr.groupby(KEYS + ["symbol"]):
        g = g.sort_values("entry_time")
        key = k[:-1]
        coin[key] = max(coin.get(key, 0.0), dd_stats(g.pnl_inr.values, g.mae_inr.values)[1])
    out = pd.DataFrame({"max_dd_all": pd.Series(comb), "max_dd_coin": pd.Series(coin)})
    out.index = pd.MultiIndex.from_tuples(out.index, names=KEYS)
    return -out                                             # shown as negative Rs, like a loss


def pool(tr, n_symbols):
    tr = tr.assign(win=tr.pnl_inr > 0, liq=tr.exit_reason == "LIQUIDATION", tpx=tr.exit_reason == "TP",
                   slx=tr.exit_reason == "SL", tmo=tr.exit_reason.isin(["TIMEOUT", "END"]))

    def agg(df, suf):
        g = df.groupby(KEYS).agg(trades=("pnl_inr", "size"), wins=("win", "sum"), pnl=("pnl_inr", "sum"))
        g["win_rate"] = g.wins / g.trades * 100
        return g.drop(columns="wins").add_suffix(suf)

    full = agg(tr, "")
    ex = tr.groupby(KEYS).agg(avg_pnl=("pnl_inr", "mean"), liq=("liq", "sum"), tp_exits=("tpx", "sum"),
                              sl_exits=("slx", "sum"), timeouts=("tmo", "sum"),
                              avg_sl_pct=("sl_pct", "mean"), avg_bars=("bars", "mean"))
    per_sym = tr.groupby(KEYS + ["symbol"]).pnl_inr.sum().gt(0).groupby(KEYS).sum().rename("coins_profit")
    pooled = full.join(ex).join(per_sym).join(drawdowns(tr))
    pooled = pooled.join(agg(tr[tr.period == "train"], "_train"), how="left").join(
        agg(tr[tr.period == "test"], "_test"), how="left")
    pooled["n_coins"] = n_symbols
    pooled = pooled.reset_index()
    pooled["_o"] = pooled.tf.map({k: i for i, k in enumerate(TF_MINUTES)})
    pooled["_s"] = (pooled.stoploss != "on").astype(int)
    return pooled.sort_values(["_s", "lev", "_o", "tp_pct"]).drop(columns=["_o", "_s"]).reset_index(drop=True)


def symbol_summary(tr):
    rows = []
    for key, g in tr.groupby(KEYS + ["symbol"], sort=False):
        g = g.sort_values("entry_time")
        closed, floating = dd_stats(g.pnl_inr.values, g.mae_inr.values)
        gains, losses = g.pnl_inr[g.pnl_inr > 0].sum(), -g.pnl_inr[g.pnl_inr <= 0].sum()
        rows.append(dict(zip(KEYS + ["symbol"], key), trades=len(g), win_rate=round((g.pnl_inr > 0).mean() * 100, 1),
                         total_pnl_inr=round(g.pnl_inr.sum(), 1), max_dd_inr=round(-closed, 1),
                         max_dd_incl_open_inr=round(-floating, 1),
                         profit_factor=round(gains / losses, 2) if losses > 0 else np.inf,
                         liquidations=int((g.exit_reason == "LIQUIDATION").sum())))
    return pd.DataFrame(rows)


def fmt(df):
    return df.round(1).to_string(index=False)


def report(pooled, a):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 2000)
    pd.set_option("display.max_columns", 50)
    cols = ["tf", "stoploss", "lev", "tp_pct", "trades", "win_rate", "pnl", "max_dd_all", "max_dd_coin", "avg_pnl",
            "liq", "sl_exits", "timeouts", "coins_profit"]
    print("\n===== POOLED RESULTS (all coins together, Rs %g margin per trade) =====" % a.margin)
    print("pnl = total Rs | avg_pnl = Rs per trade | coins_profit = coins with positive total (of %d)" % pooled.n_coins.iloc[0])
    print("max_dd_all  = max drawdown Rs, all coins combined, closed trades only")
    print("max_dd_coin = max drawdown Rs of the WORST single coin, also counting the unrealised loss of open trades")
    print("              (with stop loss OFF this is the number to trust: a trade can sink deep before it closes)")
    print(fmt(pooled[cols]))

    print("\n===== TARGET CHOSEN ON TRAIN, JUDGED ON TEST (last %d%% of each coin's data) =====" % round(a.test_frac * 100))
    rows = []
    for (tf, sl, lev), g in pooled.groupby(["tf", "stoploss", "lev"], sort=False):
        c = g[g.trades_train >= a.min_trades]
        if c.empty:
            continue
        b = c.loc[c.pnl_train.idxmax()]
        rows.append(dict(tf=tf, stoploss=sl, lev=lev, best_tp_on_train=b.tp_pct, train_trades=b.trades_train,
                         train_win=b.win_rate_train, train_pnl=b.pnl_train, test_trades=b.trades_test,
                         test_win=b.win_rate_test, test_pnl=b.pnl_test, max_dd_all_full=b.max_dd_all,
                         max_dd_coin_full=b.max_dd_coin))
    print(fmt(pd.DataFrame(rows)) if rows else "(not enough trades on train - need >= %d)" % a.min_trades)

    print("\n===== SETUPS WITH WIN RATE >= 60%% AND PROFIT (>= %d trades) =====" % a.min_trades)
    q = pooled[(pooled.win_rate >= 60) & (pooled.pnl > 0) & (pooled.trades >= a.min_trades)]
    if q.empty:
        print("none - no timeframe/target/stop-loss combination reached 60% win rate with a profit.")
    else:
        print(fmt(q[["tf", "stoploss", "lev", "tp_pct", "trades", "win_rate", "pnl", "max_dd_all", "max_dd_coin",
                     "coins_profit", "n_coins", "liq", "pnl_train", "pnl_test", "win_rate_test"]]
                  .sort_values("pnl", ascending=False).head(25)))
    print("\nCAUTION: %d combinations were tested, so some will look good by pure luck (random data also"
          " produces a few). Trust a setup only if MOST coins are profitable (coins_profit close to n_coins),"
          " pnl_train and pnl_test are BOTH positive, and liq (liquidations) is ~0. With stop loss OFF, a high"
          " win rate often hides rare very large losses." % len(pooled))


def plot(pooled, outdir, margin):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lev = pooled.lev.iloc[0]
        d = pooled[pooled.lev == lev]
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        for (tf, sl), g in d.groupby(["tf", "stoploss"], sort=False):
            ls = "-" if sl == "on" else "--"
            ax[0].plot(g.tp_pct, g.win_rate, ls, marker="o", ms=3, label=f"{tf} SL {sl}")
            ax[1].plot(g.tp_pct, g.pnl, ls, marker="o", ms=3, label=f"{tf} SL {sl}")
        ax[0].axhline(60, color="gray", lw=.8)
        ax[0].set_title("Win rate % vs target %")
        ax[1].axhline(0, color="gray", lw=.8)
        ax[1].set_title(f"Total P&L (Rs) vs target %  [{lev}x, Rs{margin:g}/trade]")
        for x in ax:
            x.set_xlabel("take-profit %")
            x.grid(alpha=.3)
        ax[1].legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "tp_curves.png"), dpi=120)
    except Exception as e:
        print("plot skipped:", e)


def main():
    ap = argparse.ArgumentParser(description="RSI divergence backtest (margin + leverage)")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results_rsi")
    ap.add_argument("--tfs", default="5m,15m,1h,4h,1d,1w")
    ap.add_argument("--rsi-len", type=int, default=14)
    ap.add_argument("--pivot", type=int, default=5, help="candles on each side that define a swing high/low")
    ap.add_argument("--min-gap", type=int, default=5, help="min candles between the two pivots")
    ap.add_argument("--max-gap", type=int, default=60, help="max candles between the two pivots")
    ap.add_argument("--min-rsi-diff", type=float, default=1.0, help="RSI must differ by at least this between pivots")
    ap.add_argument("--min-price-diff", type=float, default=0.0, help="%% by which 2nd pivot must exceed the 1st")
    ap.add_argument("--rsi-ob", type=float, default=0, help="bearish: RSI at the 1st high must be >= this (0 = off, e.g. 60)")
    ap.add_argument("--rsi-os", type=float, default=100, help="bullish: RSI at the 1st low must be <= this (100 = off, e.g. 40)")
    ap.add_argument("--trend", choices=["off", "with", "against"], default="off",
                    help="EMA trend filter: with = longs only above EMA / shorts only below; against = the opposite")
    ap.add_argument("--ema-len", type=int, default=100, help="EMA length for the trend filter (on the signal timeframe)")
    ap.add_argument("--vol-mult", type=float, default=0.0, help="volume filter: pivot-bar volume >= this x 20-bar avg (0 = off)")
    ap.add_argument("--targets", default="1,2,3,4,5,6,7,8,9,10", help="take-profit %% list, each tested separately")
    ap.add_argument("--stoploss", choices=["on", "off", "both"], default="both")
    ap.add_argument("--sl-type", choices=["swing", "atr", "pct"], default="swing")
    ap.add_argument("--sl-buffer", type=float, default=0.5, help="swing SL: ATRs beyond the divergence extreme")
    ap.add_argument("--sl-min", type=float, default=0.5, help="swing/atr SL: minimum distance %%")
    ap.add_argument("--sl-max", type=float, default=5.0, help="swing/atr SL: maximum distance %%")
    ap.add_argument("--sl-atr", type=float, default=2.0)
    ap.add_argument("--sl-pct", type=float, default=2.0)
    ap.add_argument("--max-hold", type=int, default=500, help="close the trade after this many candles")
    ap.add_argument("--margin", type=float, default=1000.0, help="Rs margin per trade")
    ap.add_argument("--leverages", default="5,10")
    ap.add_argument("--mmr", type=float, default=0.5, help="%% maintenance margin for liquidation price")
    ap.add_argument("--fee", type=float, default=0.02, help="%% per side on full position")
    ap.add_argument("--slip", type=float, default=0.02, help="%% per side")
    ap.add_argument("--tds", type=float, default=0.0)
    ap.add_argument("--test-frac", type=float, default=0.3, help="last fraction of each coin's data = test period")
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--all-trades", action="store_true", help="write trades.csv for every leverage (bigger file)")
    ap.add_argument("--no-long", action="store_true")
    ap.add_argument("--no-short", action="store_true")
    a = ap.parse_args()

    p = dict(rsi_len=a.rsi_len, pivot=a.pivot, min_gap=a.min_gap, max_gap=a.max_gap, min_rsi_diff=a.min_rsi_diff,
             min_price_diff=a.min_price_diff, rsi_ob=a.rsi_ob, rsi_os=a.rsi_os, sl_type=a.sl_type,
             sl_buffer=a.sl_buffer, sl_min=a.sl_min, sl_max=a.sl_max, sl_atr=a.sl_atr, sl_pct=a.sl_pct,
             max_hold=a.max_hold, margin=a.margin, mmr=a.mmr, fee=a.fee, slip=a.slip, tds=a.tds,
             trend=a.trend, ema_len=a.ema_len, vol_mult=a.vol_mult, allow_long=not a.no_long, allow_short=not a.no_short)
    tps = [float(x) for x in a.targets.split(",") if x.strip()]
    levs = [int(float(x)) if float(x).is_integer() else float(x) for x in a.leverages.split(",") if x.strip()]
    sl_modes = {"on": [True], "off": [False], "both": [True, False]}[a.stoploss]
    scenarios = [(tp, lev, sl) for sl in sl_modes for lev in levs for tp in tps]

    files = sorted(glob.glob(os.path.join(a.data, "*.csv")))
    if not files:
        sys.exit(f"No CSV files in {a.data}")
    os.makedirs(a.out, exist_ok=True)
    all_tr = []
    for f in files:
        sym = os.path.splitext(os.path.basename(f))[0]
        base = load_csv(f)
        bm = int(np.median(np.diff(base.index.values).astype("timedelta64[m]").astype(int)))
        print(f"[data] {sym}: {len(base)} candles ({bm}m), {base.index[0]:%Y-%m-%d} -> {base.index[-1]:%Y-%m-%d}")
        split = base.index[0] + (base.index[-1] - base.index[0]) * (1 - a.test_frac)
        all_tr += run_symbol(sym, base, [t.strip() for t in a.tfs.split(",")], p, scenarios, split)
    if not all_tr:
        sys.exit("No trades generated (not enough data or no divergences found).")
    tr = pd.DataFrame(all_tr)
    pooled = pool(tr, len(files))
    pooled.to_csv(os.path.join(a.out, "pooled.csv"), index=False)
    symbol_summary(tr).to_csv(os.path.join(a.out, "summary_by_coin.csv"), index=False)
    # leverage only scales P&L (and adds liquidations, see pooled.csv), so by default trades.csv keeps the first leverage
    (tr if a.all_trades else tr[tr.lev == tr.lev.iloc[0]]).to_csv(os.path.join(a.out, "trades.csv"), index=False)
    report(pooled, a)
    plot(pooled, a.out, a.margin)


if __name__ == "__main__":
    main()
