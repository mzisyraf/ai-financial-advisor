import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
import json


class DataProcessor:
    """Handle ETL-ish transforms + finance helpers."""

    def __init__(self):
        self.scaler = StandardScaler()

    # ───────────────────────────────── Expenses ──
    def process_expenses_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        if df is None or df.empty:
            return pd.DataFrame(), {}

        df = df.copy()
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df['bill_date'] = pd.to_datetime(df['bill_date'], errors='coerce')
        df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')
        df['month'] = df['bill_date'].dt.strftime('%Y-%m')
        df['days_until_due'] = (
            df['due_date'] - df['bill_date']).dt.days.fillna(0).astype(int)

        summary = {
            'total_rent':          df.loc[df['type'] == 'Rent',        'amount'].sum(),
            'total_electricity':   df.loc[df['type'] == 'Electricity', 'amount'].sum(),
            'total_water':         df.loc[df['type'] == 'Water',       'amount'].sum(),
            'total_ingredients':   df[df['type'].str.contains('Bakery', case=False)]['amount'].sum(),
            'total_marketing':     df[df['type'].isin(['Advertising', 'Marketing'])]['amount'].sum(),
            'total_other':         df[~df['type'].isin(['Rent', 'Electricity', 'Water']) &
                                      ~df['type'].str.contains('Bakery', case=False)]['amount'].sum(),
            'total_salary':        0  # filled later
        }

        monthly_expenses = (
            df.groupby(['month', 'type'])['amount']
              .sum().unstack(fill_value=0).reset_index()
        )
        monthly_expenses['utilities'] = (monthly_expenses.get('Electricity', 0) +
                                         monthly_expenses.get('Water', 0))
        monthly_expenses['total'] = monthly_expenses.drop(
            columns='month').sum(axis=1)
        summary['monthly_expenses'] = monthly_expenses.set_index(
            'month').to_dict('index')

        return df, summary

    # ───────────────────────────────── Sales ──
    def process_sales_data(self,
                           sales_df: pd.DataFrame,
                           products_df: pd.DataFrame | None = None
                           ) -> Tuple[pd.DataFrame, Dict]:
        if sales_df is None or sales_df.empty:
            return pd.DataFrame(), {}

        s = sales_df.copy()
        s['date'] = pd.to_datetime(s['date'], errors='coerce')

        # Parse product_sales JSON…
        def _parse(val):
            if isinstance(val, dict):
                return val
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    return json.loads(val.replace("'", '"'))
            return {}
        s['product_sales'] = s['product_sales'].apply(_parse)
        s['items_sold'] = s['product_sales'].apply(lambda d: sum(d.values()))

        if 'sale_amount' in s.columns:
            s['daily_sales'] = pd.to_numeric(
                s['sale_amount'], errors='coerce').fillna(0)
        elif products_df is not None:
            price_lut = products_df.set_index('product_id')['price'].to_dict()
            s['daily_sales'] = s['product_sales'].apply(
                lambda d: sum(int(q)*price_lut.get(int(pid), 0)
                              for pid, q in d.items())
            )
        else:
            s['daily_sales'] = s['items_sold'].astype(float)

        s['month'] = s['date'].dt.strftime('%Y-%m')

        # Monthly aggregation
        monthly = s.resample('MS', on='date').agg(
            total_sales=('daily_sales', 'sum'),
            average_daily_sales=('daily_sales', 'mean'),
            transaction_count=('daily_sales', 'count'),
            total_items_sold=('items_sold', 'sum')
        ).reset_index()
        monthly['month'] = monthly['date'].dt.strftime('%Y-%m')

        metrics = {
            'total_sales': monthly['total_sales'].sum(),
            'average_monthly_sales': monthly['total_sales'].mean(),
            'highest_sales_month': monthly.loc[monthly['total_sales'].idxmax(), 'month'],
            'lowest_sales_month':  monthly.loc[monthly['total_sales'].idxmin(), 'month']
        }

        return monthly, metrics

    # ─────────────────────────────── Employees ──
    def process_employees_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        if df is None or df.empty:
            return pd.DataFrame(), {}

        df = df.copy()
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce').fillna(0)
        df['hire_date'] = pd.to_datetime(df['hire_date'], errors='coerce')
        df['month'] = df['hire_date'].dt.strftime('%Y-%m')

        metrics = {
            'total_salary': df['salary'].sum(),
            'average_salary': df['salary'].mean(),
            'active_employees': len(df),
            'top_performer': df.loc[df['salary'].idxmax(), 'name']
        }

        monthly_salaries = df.groupby('month')['salary'].sum().to_dict()
        metrics['monthly_salaries'] = monthly_salaries
        return df, metrics

    # ─────────────────────────────── Products ──
    def process_products_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        if df is None or df.empty:
            return pd.DataFrame(), {}

        df = df.copy()
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        df['cost'] = df['price'] * 0.7
        df['profit_per_item'] = df['price'] - df['cost']
        return df, {
            'total_products': len(df),
            'average_price': df['price'].mean(),
            'average_profit_per_item': df['profit_per_item'].mean(),
            'average_profit_margin': (df['profit_per_item'] /
                                      df['price'].replace(0, np.nan)).mean()
        }

    # ─────────────────────────────── Helpers ──
    def cash_flow(self, monthly_sales: pd.DataFrame,
                  expenses_df: pd.DataFrame) -> pd.DataFrame:
        """Return monthly cash-in/out/net/cum frame."""
        inflow = monthly_sales.set_index('month')['total_sales']
        outflow = (expenses_df.groupby('month')['amount'].sum()
                   if not expenses_df.empty else pd.Series(dtype=float))
        cf = pd.DataFrame({
            'cash_in': inflow,
            'cash_out': outflow.reindex(inflow.index, fill_value=0)
        })
        cf['net_cash'] = cf['cash_in'] - cf['cash_out']
        cf['cum_cash'] = cf['net_cash'].cumsum()
        return cf.reset_index()

    def financial_ratios(self,
                         monthly_sales: pd.DataFrame,
                         exp_summary: Dict,
                         balance: Dict) -> Dict:
        sales_total = monthly_sales['total_sales'].sum() or 1
        ratios = {
            'gross_margin': (sales_total - exp_summary['total_ingredients']) / sales_total,
            'current_ratio': balance['current_assets'] / max(1, balance['current_liab']),
            'quick_ratio': ((balance['current_assets'] - balance.get('inventory', 0))
                            / max(1, balance['current_liab'])),
            'debt_to_equity': balance['total_debt'] / max(1, balance.get('equity', 1)),
            'dscr': sales_total / max(1, exp_summary['total_other']+exp_summary['total_salary'])
        }
        return ratios
