import pandas as pd
print("✅  DUMMY run_once module imported")   # already there


def run_once(cfg_path: str | None = None):
    print("✅  DUMMY run_once() called")      # << add this line

    monthly_sales = pd.DataFrame({
        "date":  [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")],
        "total_sales": [12000, 15000],
        "month": ["2024-01", "2024-02"]
    })
    expenses_df = pd.DataFrame({"month": ["2024-01", "2024-02"],
                                "amount": [9500, 10300]})
    cashflow_df = pd.DataFrame({
        "month":    monthly_sales["month"],
        "cash_in":  monthly_sales["total_sales"],
        "cash_out": expenses_df["amount"],
    })
    cashflow_df["net_cash"] = cashflow_df["cash_in"] - cashflow_df["cash_out"]
    cashflow_df["cum_cash"] = cashflow_df["net_cash"].cumsum()

    return {
        "cashflow": cashflow_df,
        "ratios":   {"gross_margin": 0.42},
        "burn_rate_months": 6.3
    }
