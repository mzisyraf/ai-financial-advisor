import requests
import json
import os
from openai import OpenAI
from jinja2 import Template


class QwenIntegration:
    """Class to integrate with Qwen 3 for AI-powered business insights"""

    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        # You can replace it with other deep thinking models as needed
        self.model_name = "qwen-plus-2025-04-28"

    def _call_qwen_api(self, prompt):
        """Call Qwen API with given prompt"""
        messages = [{"role": "user", "content": prompt}]

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            extra_body={"enable_thinking": True},
            stream=True,
        )

        answer_content = ""
        is_answering = False

        for chunk in completion:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Received content, starting to respond
            if hasattr(delta, "content") and delta.content:
                if not is_answering:
                    is_answering = True
                answer_content += delta.content

        return answer_content

    def generate_budget_plan(self, financial_data):
        """Generate personalized budget plan based on financial data"""
        prompt_template = Template("""
Given the following financial data for a small business:

- Monthly Sales: RM {{ monthly_sales }}
- Current Inventory Value: RM {{ inventory_value }}
- Employee Salaries: RM {{ salaries }} per month
- Monthly Utilities: RM {{ utilities }}

{% if monthly_expenses %}
Monthly Expense Breakdown:
{% for month in monthly_expenses | sort %}
- {{ month }}:
  - Rent: RM {{ monthly_expenses[month].rent }}
  - Utilities: RM {{ monthly_expenses[month].utilities }} (Electricity: RM {{ monthly_expenses[month].electricity }}, Water: RM {{ monthly_expenses[month].water }})
  - Ingredients: RM {{ monthly_expenses[month].ingredients }}
  - Marketing: RM {{ monthly_expenses[month].marketing }}
  - Salaries: RM {{ monthly_expenses[month].salaries }}
  - Total: RM {{ monthly_expenses[month].total }}
{% endfor %}
{% endif %}

Please provide a detailed budget plan for the next quarter (3 months) that includes:
1. Revenue projections
2. Expense management recommendations
3. Cash flow optimization strategies
4. Cost-saving measures

The output should be in clear sections with actionable recommendations.
""")

        prompt = prompt_template.render(financial_data)
        return self._call_qwen_api(prompt)

    def assess_loan_eligibility(self, business_data):
        """Assess loan eligibility based on business data"""
        prompt_template = Template("""
Evaluate the loan eligibility for a small business with the following characteristics:

- Average Monthly Sales: RM {{ avg_monthly_sales }}
- Total Assets: RM {{ total_assets }}
- Outstanding Liabilities: RM {{ liabilities }}
- Years in Business: {{ years_in_business }}
- Credit Score: {{ credit_score }}

Please provide:
1. Overall loan eligibility assessment
2. Estimated loan amount they could qualify for
3. Recommended loan terms
4. Any additional requirements or conditions

The output should be in clear sections with specific details.
""")

        prompt = prompt_template.render(business_data)
        print(prompt)
        return self._call_qwen_api(prompt)

    def analyze_financial_health(self, metrics):
        """Analyze overall financial health of the business"""
        prompt_template = Template("""
Analyze the financial health of a business based on these key metrics:

- Profit Margin: {{ profit_margin }}%
- Current Ratio: {{ current_ratio }}
- Debt-to-Equity Ratio: {{ debt_to_equity }}
- Inventory Turnover: {{ inventory_turnover }} times/month
- Employee Productivity: RM {{ employee_productivity }} sales/employee

Provide a comprehensive analysis including:
1. Strengths and weaknesses
2. Trend analysis
3. Recommendations for improvement
4. Risk assessment

The output should be in clear sections with specific details.
""")

        prompt = prompt_template.render(metrics)
        print(prompt)
        return self._call_qwen_api(prompt)
