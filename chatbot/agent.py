from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage            # NEW
import os
from typing import Dict, Any


def build(metrics: Dict[str, Any]):
    """
    Returns an agent with two tools and a finance-focused system prompt.
    """

    # ── tool wrappers ───────────────────────────────────────
    def get_metric(name: str):
        val = metrics.get(name)
        return float(val) if isinstance(val, (int, float)) else f"{val}"

    def get_table(name: str):
        tbl = metrics.get(name)
        return tbl.head(10).to_markdown(index=False) if hasattr(tbl, "head") else "Table not found"

    tools = [
        Tool(
            name="get_metric",
            func=get_metric,
            description="Return numeric KPI by name, e.g. 'current_ratio'."
        ),
        Tool(
            name="get_table",
            func=get_table,
            description="Preview a data table such as 'cashflow' or 'sales_monthly'."
        ),
    ]

    # ── LLM backend ─────────────────────────────────────────
    llm = ChatOpenAI(
        model_name="qwen-plus-2025-04-28",
        openai_api_key=os.getenv("QWEN_API_KEY"),
        openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        temperature=0.2,
    )

    # ── SYSTEM PROMPT ───────────────────────────────────────
    # tailor this text however you like
    sys_prompt = SystemMessage(content="""
    You are **MSME Finance Copilot**, a concise financial-insight assistant.
    • ALWAYS rely on the provided tools (`get_metric`, `get_table`) to retrieve
    numbers; do not invent values.
    • Focus on practical advice: cash-flow projections, loan eligibility, budgeting,
    break-even analysis, and KPI interpretation.
    • Respond with clear Markdown bullets or short paragraphs.
    """.strip())

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
        agent_kwargs={"system_message": sys_prompt},  # ⬅︎ here
    )
