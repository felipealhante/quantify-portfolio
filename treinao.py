import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def normalize_ticker(t):
    return t.strip().upper().replace("ˆ", "^")


def main():
    tickers_input = input("Enter tickers separated by commas (e.g., AAPL,MSFT,NVDA): ")
    values_input = input("Enter dollar value for each asset (e.g., 40000,30000,30000): ")

    tickers = [normalize_ticker(t) for t in tickers_input.split(",")]
    values = np.array([float(v) for v in values_input.split(",")])

    if len(tickers) != len(values):
        raise ValueError("Number of tickers and values must match.")

    total_value = values.sum()
    weights = values / total_value

    print("\nPortfolio Holdings:")
    for t, v, w in zip(tickers, values, weights):
        print(f"{t}: ${v:,.2f}  |  Weight: {w*100:.2f}%")

    print(f"\nTotal Portfolio Value: ${total_value:,.2f}")

    # ----------------------------
    # Pie Chart — Weights
    # ----------------------------
    plt.figure(figsize=(8, 8))
    plt.pie(
        weights,
        labels=tickers,
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Portfolio Weights by Asset (%)")
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Bar Chart — Weights
    # ----------------------------
    plt.figure(figsize=(10, 6))
    plt.bar(tickers, weights * 100)
    plt.ylabel("Weight (%)")
    plt.title("Portfolio Asset Weights")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
