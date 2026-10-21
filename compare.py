#!/usr/bin/env python3
"""Side-by-side comparison of two result folders (each has pooled.csv).
   python compare.py results_base results_trend [--lev 5]"""
import sys
import pandas as pd

a = sys.argv[1]; b = sys.argv[2]
lev = int(sys.argv[sys.argv.index("--lev") + 1]) if "--lev" in sys.argv else 5
k = ["tf", "stoploss", "lev", "tp_pct"]
cols = ["trades", "win_rate", "pnl", "pnl_train", "pnl_test", "coins_profit", "max_dd_coin"]
A = pd.read_csv(f"{a}/pooled.csv").set_index(k)[cols].add_suffix("_A")
B = pd.read_csv(f"{b}/pooled.csv").set_index(k)[cols].add_suffix("_B")
m = A.join(B, how="outer").reset_index()
m = m[m.lev == lev]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(f"A = {a} | B = {b} | leverage {lev}x\n")
print(m[["tf","stoploss","tp_pct","trades_A","trades_B","pnl_A","pnl_B","pnl_test_A","pnl_test_B","coins_profit_A","coins_profit_B"]].round(0).to_string(index=False))
print("\nSL ON, summary per timeframe (sum over all TPs):")
g = m[m.stoploss == "on"].groupby("tf")[["pnl_A","pnl_B","pnl_test_A","pnl_test_B"]].sum().round(0)
print(g)
print("\nB setups with pnl>0 on BOTH train and test, >=30 trades, >=4 coins profitable:")
q = m[(m.pnl_train_B > 0) & (m.pnl_test_B > 0) & (m.trades_B >= 30) & (m.coins_profit_B >= 4)]
print(q[["tf","stoploss","tp_pct","trades_B","win_rate_B","pnl_B","pnl_train_B","pnl_test_B","coins_profit_B"]].round(1).to_string(index=False) if len(q) else "none")
