import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json


class DataProcessor:
    """Class to process and prepare data for visualization and AI insights"""

    def __init__(self):
        self.scaler = StandardScaler()

    def process_expenses_data(self, expenses_df):
        print("\n========== Processing Expenses Data ==========")

        if expenses_df is None or expenses_df.empty:
            print("Warning: Empty or None expenses data received")
            return pd.DataFrame(), {}

        print(f"Initial expenses data shape: {expenses_df.shape}")
        print("Columns:", expenses_df.columns.tolist())

        # Fill missing values and ensure correct data types
        expenses_df['amount'] = pd.to_numeric(
            expenses_df['amount'], errors='coerce').fillna(0)
        expenses_df['bill_date'] = pd.to_datetime(
            expenses_df['bill_date'], errors='coerce')
        expenses_df['due_date'] = pd.to_datetime(
            expenses_df['due_date'], errors='coerce')

        # Create new features
        expenses_df['days_until_due'] = (
            expenses_df['due_date'] - expenses_df['bill_date']).dt.days
        expenses_df['is_utility'] = ~expenses_df['type'].str.contains(
            'Bakery', case=False)
        expenses_df['days_until_due'] = expenses_df['days_until_due'].fillna(
            0).astype(int)

        print("Sample of processed expenses data:\n", expenses_df.head())

        categorized_expenses = {
            'total_rent': expenses_df[expenses_df['type'] == 'Rent']['amount'].sum(),
            'total_electricity': expenses_df[expenses_df['type'] == 'Electricity']['amount'].sum(),
            'total_water': expenses_df[expenses_df['type'] == 'Water']['amount'].sum(),
            'total_ingredients': expenses_df[expenses_df['type'].str.contains('Bakery - ', case=False)]['amount'].sum(),
            'total_other': expenses_df[~expenses_df['type'].isin(['Rent', 'Electricity', 'Water']) & ~expenses_df['type'].str.contains('Bakery', case=False)]['amount'].sum(),
            'total_salary': 0
        }

        print("Categorized expenses:\n", categorized_expenses)

        if 'bill_date' in expenses_df.columns and not expenses_df.empty:
            expenses_df['month'] = expenses_df['bill_date'].dt.strftime(
                '%Y-%m')

            monthly_rent = expenses_df[expenses_df['type'] == 'Rent'].groupby('month')[
                'amount'].sum().to_dict()
            monthly_electricity = expenses_df[expenses_df['type'] == 'Electricity'].groupby(
                'month')['amount'].sum().to_dict()
            monthly_water = expenses_df[expenses_df['type'] == 'Water'].groupby('month')[
                'amount'].sum().to_dict()
            monthly_ingredients = expenses_df[expenses_df['type'].str.contains(
                'Bakery - ', case=False)].groupby('month')['amount'].sum().to_dict()
            monthly_marketing = expenses_df[expenses_df['type'].isin(
                ['Advertising', 'Marketing'])].groupby('month')['amount'].sum().to_dict()

            all_months = set(monthly_rent.keys()) | set(monthly_electricity.keys()) | set(
                monthly_water.keys()) | set(monthly_ingredients.keys()) | set(monthly_marketing.keys())

            monthly_expenses = {}
            for month in sorted(all_months):
                utilities = monthly_electricity.get(
                    month, 0) + monthly_water.get(month, 0)
                monthly_expenses[month] = {
                    'rent': monthly_rent.get(month, 0),
                    'utilities': utilities,
                    'electricity': monthly_electricity.get(month, 0),
                    'water': monthly_water.get(month, 0),
                    'ingredients': monthly_ingredients.get(month, 0),
                    'marketing': monthly_marketing.get(month, 0),
                    'total': monthly_rent.get(month, 0) + utilities + monthly_ingredients.get(month, 0) + monthly_marketing.get(month, 0)
                }

            print("Monthly expenses summary:\n", monthly_expenses)
        else:
            monthly_expenses = {}
            print("No valid bill_date found for monthly grouping.")

        return expenses_df, {
            'categorized_expenses': categorized_expenses,
            'monthly_expenses': monthly_expenses
        }

    def process_sales_data(self, sales_df, products_df=None):
        """
        Process daily sales:
          - sales_df must contain 'date', 'product_sales' (dict or JSON string) 
            and optionally 'sale_amount'.
          - products_df (optional): DataFrame with ['product_id','price'] 
            to compute revenue if 'sale_amount' is missing.
        Returns: 
          monthly_df, monthly_metrics
        """
        print("\n========== Processing Sales Data ==========")

        if sales_df is None or sales_df.empty:
            print("Warning: Empty or None sales data received")
            return pd.DataFrame(), {}

        # 1) Date conversion
        sales_df['date'] = pd.to_datetime(sales_df['date'], errors='coerce')

        # 2) Parse product_sales into a dict of {id_str: qty_int}
        def parse_product_sales(val):
            if isinstance(val, dict):
                return val
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except json.JSONDecodeError:
                    try:
                        return json.loads(val.replace("'", '"'))
                    except Exception:
                        print(f"Failed to parse product_sales: {val!r}")
            return {}

        sales_df['product_sales'] = sales_df['product_sales'].apply(
            parse_product_sales)

        # 3) Compute total items sold per day
        sales_df['items_sold'] = sales_df['product_sales'].apply(
            lambda d: sum(d.values()))

        # 4) Compute daily_sales (revenue):
        if 'sale_amount' in sales_df.columns:
            # use the DB-provided revenue if available
            sales_df['daily_sales'] = pd.to_numeric(
                sales_df['sale_amount'], errors='coerce'
            ).fillna(0)
        elif products_df is not None:
            # build a price lookup: { product_id (int) → price (float) }
            price_lookup = products_df.set_index(
                'product_id')['price'].to_dict()

            def compute_revenue(d):
                rev = 0.0
                for pid_str, qty in d.items():
                    try:
                        pid = int(pid_str)
                        rev += qty * price_lookup.get(pid, 0.0)
                    except ValueError:
                        continue
                return rev
            sales_df['daily_sales'] = sales_df['product_sales'].apply(
                compute_revenue)
        else:
            # fallback: treat each item as “1 unit currency” if no info
            sales_df['daily_sales'] = sales_df['items_sold'].astype(float)

        # 5) Average price per item = revenue ÷ quantity
        sales_df['avg_price_per_item'] = (
            sales_df['daily_sales'] / sales_df['items_sold']
        ).replace([np.inf, -np.inf], 0).fillna(0)

        # 6) Time-based features
        sales_df['day_of_week'] = sales_df['date'].dt.day_name().fillna('Unknown')
        sales_df['month'] = sales_df['date'].dt.strftime(
            '%Y-%m').fillna('Unknown')

        print("Sample of processed sales data:\n", sales_df.head())

        # 7) Monthly aggregation
        monthly_df = sales_df.resample('ME', on='date').agg({
            'daily_sales': ['sum', 'mean', 'count'],
            'items_sold': 'sum'
        }).reset_index()
        monthly_df.columns = [
            'date', 'total_sales', 'average_daily_sales', 'transaction_count', 'total_items_sold'
        ]
        monthly_df['month'] = monthly_df['date'].dt.strftime(
            '%Y-%m').fillna('Unknown')

        print("Monthly aggregated sales:\n", monthly_df.head())

        # 8) High-level metrics
        monthly_metrics = {
            'total_sales':           monthly_df['total_sales'].sum(),
            'average_monthly_sales': monthly_df['total_sales'].mean(),
            'highest_sales_month':   monthly_df.loc[monthly_df['total_sales'].idxmax()]['month']
            if not monthly_df.empty else None,
            'lowest_sales_month':    monthly_df.loc[monthly_df['total_sales'].idxmin()]['month']
            if not monthly_df.empty else None
        }

        print("Sales metrics:\n", monthly_metrics)

        return monthly_df, monthly_metrics

    def process_monthly_summary(self, monthly_df):
        print("\n========== Processing Monthly Summary ==========")

        if monthly_df is None or monthly_df.empty:
            print("Warning: Empty or None monthly data received")
            return pd.DataFrame(), {}

        monthly_df['total_sales'] = pd.to_numeric(
            monthly_df['total_sales'], errors='coerce').fillna(0)
        monthly_df['month'] = monthly_df['month'].astype(str)

        print("Monthly summary sample:\n", monthly_df.head())

        metrics = {
            'total_sales': monthly_df['total_sales'].sum(),
            'average_monthly_sales': monthly_df['total_sales'].mean(),
            'best_month': monthly_df.loc[monthly_df['total_sales'].idxmax()]['month'] if not monthly_df.empty else None,
            'worst_month': monthly_df.loc[monthly_df['total_sales'].idxmin()]['month'] if not monthly_df.empty else None,
            'sales_trend': 'Upward' if len(monthly_df) > 1 and monthly_df.iloc[-1]['total_sales'] > monthly_df.iloc[0]['total_sales'] else 'Downward' if len(monthly_df) > 1 else 'Neutral'
        }

        print("Monthly summary metrics:\n", metrics)

        return monthly_df, metrics

    def process_employees_data(self, employees_df):
        print("\n========== Processing Employees Data ==========")

        if employees_df is None or employees_df.empty:
            print("Warning: Empty or None employees data received")
            return pd.DataFrame(), {}

        employees_df['salary'] = pd.to_numeric(
            employees_df['salary'], errors='coerce').fillna(0)
        employees_df['hire_date'] = pd.to_datetime(
            employees_df['hire_date'], errors='coerce')

        employee_metrics = {
            'total_salary': employees_df['salary'].sum(),
            'average_salary': employees_df['salary'].mean(),
            'active_employees': len(employees_df),
            'top_performer': employees_df.loc[employees_df['salary'].idxmax()]['name'] if not employees_df.empty else 'N/A'
        }

        print("Employee metrics:\n", employee_metrics)

        monthly_salaries = {}
        if 'hire_date' in employees_df.columns and not employees_df.empty:
            employees_df['month'] = employees_df['hire_date'].dt.strftime(
                '%Y-%m')
            monthly_salaries = employees_df.groupby(
                'month')['salary'].sum().to_dict()

        return employees_df, {
            'employee_metrics': employee_metrics,
            'monthly_salaries': monthly_salaries
        }

    def process_products_data(self, products_df):
        print("\n========== Processing Products Data ==========")

        if products_df is None or products_df.empty:
            print("Warning: Empty or None products data received")
            return pd.DataFrame(), {}

        products_df['price'] = pd.to_numeric(
            products_df['price'], errors='coerce').fillna(0)
        if 'category' not in products_df.columns:
            products_df['category'] = 'Unknown'

        products_df['cost'] = products_df['price'] * 0.7
        products_df['profit_per_item'] = products_df['price'] - \
            products_df['cost']

        print("Processed product data:\n", products_df.head())

        return products_df, {
            'total_products': len(products_df),
            'average_price': products_df['price'].mean(),
            'average_profit_per_item': products_df['profit_per_item'].mean()
        }
