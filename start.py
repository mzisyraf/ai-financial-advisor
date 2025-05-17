"""Main application entry point for Alibaba Cloud Business Dashboard"""

import argparse
import os
import sys
from data_extraction.rds_connector import RDSConnector
from data_processing.data_processor import DataProcessor
from ai_insights.qwen_integration import QwenIntegration
import json


def main():
    """Main function to run the business dashboard application"""
    parser = argparse.ArgumentParser(
        description='Alibaba Cloud Business Dashboard')
    parser.add_argument('--config', type=str,
                        help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        print("Error: Configuration file not found")
        sys.exit(1)

    # Initialize components
    rds_connector = RDSConnector(
        host=config['rds']['host'],
        port=config['rds']['port'],
        user=config['rds']['user'],
        password=config['rds']['password'],
        database=config['rds']['database']
    )

    data_processor = DataProcessor()
    qwen_integration = QwenIntegration(
        api_key=config['qwen']['api_key']
    )

    try:
        # Connect to RDS
        rds_connector.connect()

        # Extract data
        expenses_df = rds_connector.extract_expenses_data()

        # Process expenses
        _, categorized_expenses = data_processor.process_expenses_data(
            expenses_df)

        # Extract and process sales data
        sales_df = rds_connector.extract_sales_data()
        monthly_sales_df, sales_metrics = data_processor.process_sales_data(
            sales_df)

        # Extract other data
        employees_df = rds_connector.extract_employees_data()
        products_df, product_metrics = data_processor.process_products_data(
            rds_connector.extract_products_data())

        # Get monthly summary (use derived data if needed)
        monthly_df = rds_connector.extract_monthly_summary()
        if monthly_df.empty and not monthly_sales_df.empty:
            monthly_df = monthly_sales_df

        # Generate AI insights
        print("\n=== Budget Planning Recommendation ===")

        # Combine all financial data for Qwen
        financial_data = {
            "monthly_sales": sales_metrics.get('average_monthly_sales', 0),
            # Using ingredient costs as inventory proxy
            "inventory_value": categorized_expenses.get('total_ingredients', 0),
            "salaries": categorized_expenses.get('total_salary', 0),
            "utilities": categorized_expenses.get('total_rent', 0) + categorized_expenses.get('total_electricity', 0) + categorized_expenses.get('total_water', 0)
        }

        # Add monthly breakdown if available
        if hasattr(expenses_df, 'monthly_expenses') and expenses_df.monthly_expenses:
            financial_data['monthly_expenses'] = expenses_df.monthly_expenses

        if employees_df is not None and hasattr(employees_df, 'monthly_salaries') and employees_df.monthly_salaries:
            financial_data['monthly_salaries'] = employees_df.monthly_salaries

            # Merge salary data into monthly expenses
            for month, salary in employees_df.monthly_salaries.items():
                if month in financial_data['monthly_expenses']:
                    financial_data['monthly_expenses'][month]['salaries'] = salary
                    financial_data['monthly_expenses'][month]['total'] += salary
                else:
                    financial_data['monthly_expenses'][month] = {
                        'salaries': salary,
                        'rent': 0,
                        'utilities': 0,
                        'electricity': 0,
                        'water': 0,
                        'ingredients': 0,
                        'marketing': 0,
                        'total': salary
                    }

        budget_plan = qwen_integration.generate_budget_plan(financial_data)
        print(budget_plan)

        print("\n=== Loan Eligibility Assessment ===")
        business_data = {
            "avg_monthly_sales": sales_metrics.get('average_monthly_sales', 0),
            # Inventory value
            "total_assets": categorized_expenses.get('total_ingredients', 0),
            # Unpaid bills
            "liabilities": categorized_expenses.get('total_other', 0),
            "years_in_business": 3,  # Example value - should be retrieved from data
            "credit_score": 720  # Example value - should be retrieved from data
        }
        loan_assessment = qwen_integration.assess_loan_eligibility(
            business_data)
        print(loan_assessment)

        print("\n=== Financial Health Analysis ===")
        metrics = {
            "profit_margin": product_metrics.get('average_profit_margin', 0),
            "current_ratio": 1.8,  # Example value - should be calculated from real data
            "debt_to_equity": 0.6,  # Example value - should be calculated from real data
            "menu_profit_margin": product_metrics.get('average_profit_margin', 0),
            "employee_productivity": sales_metrics.get('average_daily_sales', 0) / len(employees_df) if employees_df is not None and len(employees_df) > 0 else 0
        }
        financial_analysis = qwen_integration.analyze_financial_health(metrics)
        print(financial_analysis)

    finally:
        # Clean up
        rds_connector.close_connection()


if __name__ == '__main__':
    main()
