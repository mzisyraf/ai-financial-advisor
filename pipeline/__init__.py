from data_extraction.rds_connector import RDSConnector
from data_processing.data_processor import DataProcessor
from config.config import load


def run_once(cfg_path: str | None = None):
    cfg = load(cfg_path)
    rds = RDSConnector(**cfg['rds'])
    rds.connect()

    dp = DataProcessor()

    # ── extraction ──────────────────
    expenses_df = rds.extract_expenses_data()
    sales_df = rds.extract_sales_data()
    products_df = rds.extract_products_data()
    employees_df = rds.extract_employees_data()

    # ── processing ──────────────────
    expenses_df, exp_summary = dp.process_expenses_data(expenses_df)
    products_df, prod_metrics = dp.process_products_data(products_df)
    monthly_sales, sales_mets = dp.process_sales_data(sales_df, products_df)
    if monthly_sales.empty:
        # guarantee required columns exist
        monthly_sales = (
            pd.DataFrame({"date": [], "total_sales": [], "month": []})
        )
    employees_df, emp_metrics = dp.process_employees_data(employees_df)

    exp_summary['total_salary'] = emp_metrics['total_salary']

    cashflow_df = dp.cash_flow(monthly_sales, expenses_df)
    if cashflow_df.empty:
        cashflow_df = pd.DataFrame(
            {"month": [], "cash_in": [], "cash_out": [],
             "net_cash": [], "cum_cash": []}
        )
    ratios = dp.financial_ratios(monthly_sales, exp_summary, {
        "current_assets": 85000,
        "current_liab":  32000,
        "total_debt":    15000,
        "equity":        53000,
        "inventory":     exp_summary['total_ingredients']
    })

    rds.close_connection()

    return {
        'expenses_df': expenses_df,
        'sales_monthly': monthly_sales,
        'products_df': products_df,
        'employees_df': employees_df,
        'cashflow': cashflow_df,
        'ratios': ratios,
        'burn_rate_months': (cashflow_df['cum_cash'].iloc[-1] /
                             -cashflow_df['net_cash'].iloc[-1]
                             if cashflow_df['net_cash'].iloc[-1] < 0 else 99)
    }
