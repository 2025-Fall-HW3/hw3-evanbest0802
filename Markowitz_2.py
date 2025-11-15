"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=50, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        # 只投資於非 exclude 的資產（例如 11 個 sector ETF）
        assets = self.price.columns[self.price.columns != self.exclude]
        dates = self.price.index

        # 參數設定：動能視窗、均線視窗、一次押幾檔 sector
        mom_window = 252   # 約一年，用來看動能 (momentum)
        ma_window = 200    # 約 200 日均線，用來做趨勢濾網
        top_k = 2          # 每次只押前 2 名 sector

        for i, date in enumerate(dates):

            # 暖身期：資料不夠長，先用等權重（全部 sector 等權）
            if i < max(mom_window, ma_window):
                w_eq = np.ones(len(assets)) / len(assets)
                self.portfolio_weights.loc[date, assets] = pd.Series(w_eq, index=assets)
                continue

            # ===== 1. SPY 趨勢濾網：判斷 risk-on / risk-off =====
            # 1-1. SPY 近一年累積報酬
            spy_ret_window = self.returns["SPY"].iloc[i - mom_window : i]
            spy_cum_ret = (1.0 + spy_ret_window).prod() - 1.0

            # 1-2. SPY 是否站上 200 日均線
            spy_price_window = self.price["SPY"].iloc[i - ma_window : i]
            spy_ma = spy_price_window.mean()
            spy_current = self.price["SPY"].iloc[i]

            risk_on = (spy_cum_ret > 0.0) and (spy_current > spy_ma)

            if not risk_on:
                # Risk-off：全部退到現金（所有 sector 權重 = 0）
                self.portfolio_weights.loc[date, assets] = 0.0
                continue

            # ===== 2. Risk-on 時做 Sector Dual Momentum =====
            # 2-1. 計算每個 sector 過去 mom_window 天的累積報酬
            R_mom = self.returns[assets].iloc[i - mom_window : i]
            cum_ret = (1.0 + R_mom).prod() - 1.0   # Series, index=assets

            # 只保留正動能的 sector，並依動能降冪排序
            winners = cum_ret[cum_ret > 0.0].sort_values(ascending=False)

            if len(winners) > 0:
                # 2-2. 選出前 top_k 名贏家
                selected = winners.head(top_k).index

                # 2-3. 在較短視窗內估計波動度，用來做 inverse volatility 權重
                R_sel = self.returns[selected].iloc[i - ma_window : i]
                sigma_sel = R_sel.std(ddof=1).replace(0, np.nan)

                inv_vol_sel = 1.0 / sigma_sel
                inv_vol_sel = inv_vol_sel.replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(0.0)

                if inv_vol_sel.sum() == 0:
                    # 如果全部標的的波動都算不出來，就在 selected 裡等權
                    w_sel = pd.Series(
                        np.ones(len(selected)) / len(selected),
                        index=selected,
                    )
                else:
                    w_sel = inv_vol_sel / inv_vol_sel.sum()

                # 把 selected 的權重放進完整 assets 的權重向量，其餘為 0
                w_full = pd.Series(0.0, index=assets)
                w_full.loc[selected] = w_sel.values

            else:
                # 沒有任何 sector 有正動能：
                # 改用所有 sector 做 inverse volatility（類 risk parity）
                R_vol = self.returns[assets].iloc[i - ma_window : i]
                sigma = R_vol.std(ddof=1).replace(0, np.nan)

                inv_vol = 1.0 / sigma
                inv_vol = inv_vol.replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(0.0)

                if inv_vol.sum() == 0:
                    w_full = pd.Series(
                        np.ones(len(assets)) / len(assets),
                        index=assets,
                    )
                else:
                    w_full = inv_vol / inv_vol.sum()

            # 寫入當日權重（只對 assets；SPY 權重保持 NaN，稍後會被 fillna(0) 變成 0）
            self.portfolio_weights.loc[date, assets] = w_full
        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
