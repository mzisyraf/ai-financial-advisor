from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
import os
from typing import Dict


def build(ratios: Dict):
    """
    Returns a LangChain agent that can pull live metrics via get_metric().
    Supply *ratios* dict each time you rebuild the agent (easiest).
    """

    def get_metric(name: str):
        return ratios.get(name, "Metric not available")

    tools = [
        Tool(
            name="get_metric",
            func=get_metric,
            description="Return the latest finance ratio given its name."
        )
    ]
    # uses env var OPENAI_API_KEY
    llm = ChatOpenAI(
        model_name="qwen-plus-2025-04-28",
        openai_api_key=os.getenv("QWEN_API_KEY"),
        openai_api_base="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        temperature=0.2,
    )
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False
    )
    return agent
