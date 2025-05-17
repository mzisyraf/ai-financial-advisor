import psycopg2
import pandas as pd
import json
from sqlalchemy import create_engine
from datetime import datetime


class RDSConnector:
    """Class to connect to Alibaba Cloud RDS and extract data"""

    def __init__(self, host, port, user, password, database):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.engine = None

        # Create SQLAlchemy engine URL
        self.engine_url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        self.engine = create_engine(self.engine_url)

    def connect(self):
        """Establish connection to RDS"""
        print(
            f"Attempting to connect to RDS: {self.host}:{self.port}/{self.database}")
        try:
            self.connection = self.engine.connect()
            print("Successfully connected to RDS using SQLAlchemy")
        except Exception as e:
            print(f"Error connecting to RDS: {e}")

    def extract_expenses_data(self):
        """Extract expenses data (utilities + stock purchases)"""
        print("Extracting expenses data...")
        if self.connection is None:
            print("ERROR: No database connection established")
            return None

        query = "SELECT * FROM expenses WHERE status = 'Paid'"
        try:
            result = pd.read_sql(query, self.engine)
            print(f"Retrieved {len(result)} expense records")
            print("Sample of expenses data:\n", result.head())

            # Ensure correct data types
            if not result.empty:
                result['amount'] = pd.to_numeric(
                    result['amount'], errors='coerce').fillna(0)
                result['bill_date'] = pd.to_datetime(
                    result['bill_date'], errors='coerce')
                result['due_date'] = pd.to_datetime(
                    result['due_date'], errors='coerce')

            return result
        except Exception as e:
            print(f"Error extracting expenses data: {e}")
            return None

    def extract_sales_data(self):
        """Extract daily sales data, preserving JSONB in `product_sales`."""
        print("Extracting sales data...")
        if self.connection is None:
            print("ERROR: No database connection established")
            return None

        query = "SELECT * FROM daily_sales"
        try:
            result = pd.read_sql(query, self.engine)
            print(f"Retrieved {len(result)} sales records")
            print("Sample of daily sales data:\n", result.head())

            if not result.empty:
                # Ensure 'date' is datetime
                result['date'] = pd.to_datetime(
                    result['date'], errors='coerce')

                # Helper to normalize JSONB / JSON strings into valid JSON text
                def fix_product_sales(value):
                    # If SQLAlchemy already parsed it as a dict, dump it straight back to JSON text
                    if isinstance(value, dict):
                        return json.dumps(value)

                    # If it’s a JSON string, try to load & re-dump it
                    if isinstance(value, str):
                        try:
                            data = json.loads(value)
                        except json.JSONDecodeError:
                            # Fallback: replace single quotes with doubles and retry
                            try:
                                data = json.loads(value.replace("'", '"'))
                            except Exception:
                                print(
                                    f"Failed to parse product_sales: {value!r}")
                                data = {}
                        return json.dumps(data)

                    # Everything else → empty object
                    return json.dumps({})

                # Apply the fix
                result['product_sales'] = result['product_sales'].apply(
                    fix_product_sales)

                # If your table uses `sale_amount` instead of `daily_sales`, rename it:
                if 'daily_sales' not in result.columns and 'sale_amount' in result.columns:
                    result['daily_sales'] = pd.to_numeric(
                        result['sale_amount'], errors='coerce'
                    ).fillna(0)

            return result

        except Exception as e:
            print(f"Error extracting sales data: {e}")
            return None

    def extract_monthly_summary(self):
        """Extract monthly aggregated data"""
        print("Extracting monthly summary data...")
        if self.connection is None:
            print("ERROR: No database connection established")
            return None

        query = "SELECT * FROM monthly_sales"
        try:
            result = pd.read_sql(query, self.engine)
            print(f"Retrieved {len(result)} monthly records")
            print("Sample of monthly sales data:\n", result.head())

            # Ensure correct data types
            if not result.empty:
                result['total_sales'] = pd.to_numeric(
                    result['total_sales'], errors='coerce').fillna(0)
                result['month'] = result['month'].astype(str)

            return result
        except Exception as e:
            print(f"Error extracting monthly data: {e}")
            return None

    def extract_employees_data(self):
        """Extract employees data"""
        print("Extracting employees data...")
        if self.connection is None:
            print("ERROR: No database connection established")
            return None

        query = "SELECT * FROM employees WHERE is_active = 1"
        try:
            result = pd.read_sql(query, self.engine)
            print(f"Retrieved {len(result)} active employees")
            print("Sample of employees data:\n", result.head())

            # Ensure correct data types
            if not result.empty:
                result['salary'] = pd.to_numeric(
                    result['salary'], errors='coerce').fillna(0)
                result['hire_date'] = pd.to_datetime(
                    result['hire_date'], errors='coerce')

            return result
        except Exception as e:
            print(f"Error extracting employees data: {e}")
            return None

    def extract_products_data(self):
        """Extract products items data"""
        print("Extracting products data...")
        if self.connection is None:
            print("ERROR: No database connection established")
            return None

        query = "SELECT * FROM products"
        try:
            result = pd.read_sql(query, self.engine)
            print(f"Retrieved {len(result)} product items")
            print("Sample of products data:\n", result.head())

            # Ensure correct data types
            if not result.empty:
                result['price'] = pd.to_numeric(
                    result['price'], errors='coerce').fillna(0)

            return result
        except Exception as e:
            print(f"Error extracting products data: {e}")
            return None

    def close_connection(self):
        """Close RDS connection"""
        if self.connection:
            self.connection.close()
            print("Connection closed")
