import argparse
from pipeline import run_once
from pprint import pprint
from ai_insights.qwen_integration import QwenIntegration
import os
import math

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="optional JSON config path")
    args = p.parse_args()

    result = run_once(args.config)
    print("\n── Cash-flow (tail) ──")
    print(result['cashflow'].tail())
    print("\n── Ratios ──")
    pprint(result['ratios'])

    # # ── Qwen insights ─────────────────────────
    # qi = QwenIntegration(api_key=os.getenv("QWEN_API_KEY"))

    # plan = qi.generate_budget_plan({
    #     "monthly_sales": result["sales_monthly"]["total_sales"].mean(),
    #     "inventory_value": result["expenses_df"]["amount"].sum(),
    #     "salaries": 0,
    #     "utilities": 0
    # })
    # print("\n── Budget plan (Qwen) ──\n", plan)
