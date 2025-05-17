import os
import json
import textwrap
from openai import OpenAI
from jinja2 import Template


class QwenIntegration:
    """
    Wrapper around Qwen-Plus (compatible-mode) that produces
    *concise, bullet-style* answers for:
      • Budget plan
      • Loan eligibility
      • Financial-health check
    """

    MODEL_NAME = "qwen-plus-2025-04-28"
    BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    # ───────────────────────────────────────────────
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.getenv("QWEN_API_KEY") or ""
        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL)

    # ───────────────────────────────────────────────
    # low-level helper
    def _run(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=messages,
            temperature=0.3,
            max_tokens=512,          # keep it short
            extra_body={"enable_thinking": True},
            stream=False
        )
        return resp.choices[0].message.content.strip()

    # ───────────────────────────────────────────────
    # 1) BUDGET PLAN
    def generate_budget_plan(self, data: dict) -> str:
        prompt = Template(textwrap.dedent("""\
            **Context**
            • Avg monthly sales : RM {{ monthly_sales | int }}
            • Inventory value   : RM {{ inventory_value | int }}
            • Salaries / month  : RM {{ salaries | int }}
            • Utilities / month : RM {{ utilities | int }}

            **Task**  
            Draft a 3-month budget plan:
            1. Revenue goal
            2. Spending cap (rent, utilities, salaries, COGS, marketing)
            3. Net cash target
            Respond in ≤120 words, *Markdown bullets only*.
        """)).render(**data)

        messages = [
            {"role": "system",
             "content":
             "You are a succinct SME-finance advisor. "
             "Reply with brief Markdown bullet lists only."},
            {"role": "user", "content": prompt}
        ]
        return self._run(messages)

    # ───────────────────────────────────────────────
    # 2) LOAN ELIGIBILITY
    def assess_loan_eligibility(self, biz: dict) -> str:
        prompt = Template(textwrap.dedent("""\
            • Avg monthly sales : RM {{ avg_monthly_sales | int }}
            • Assets            : RM {{ total_assets | int }}
            • Liabilities       : RM {{ liabilities | int }}
            • Years operating   : {{ years_in_business }}
            • Credit score      : {{ credit_score }}

            Evaluate loan eligibility and suggest:
            • Max loan amount
            • Ideal term & rate
            • Key approval risks
            ≤80 words, Markdown bullets.
        """)).render(**biz)

        messages = [
            {"role": "system",
             "content":
             "You are a bank credit analyst speaking to a micro-business owner. "
             "Be clear and concise; use Markdown bullets only."},
            {"role": "user", "content": prompt}
        ]
        return self._run(messages)

    # ───────────────────────────────────────────────
    # 3) FINANCIAL HEALTH
    def analyze_financial_health(self, m: dict) -> str:
        prompt = Template(textwrap.dedent("""\
            • Profit margin         : {{ profit_margin }} %
            • Current ratio         : {{ current_ratio }}
            • Debt-to-equity ratio  : {{ debt_to_equity }}
            • Inventory turnover    : {{ inventory_turnover }}×/month
            • Sales per employee    : RM {{ employee_productivity | int }}

            Give:
            • 2 strengths
            • 2 weaknesses
            • 2 quick wins
            • Overall health score /10
            Respond in ≤120 words, Markdown bullets.
        """)).render(**m)

        messages = [
            {"role": "system",
             "content":
             "You are an MSME financial coach. "
             "Answer with short Markdown bullet lists; no extra commentary."},
            {"role": "user", "content": prompt}
        ]
        return self._run(messages)
