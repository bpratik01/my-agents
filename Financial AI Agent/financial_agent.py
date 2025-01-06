from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.googlesearch import GoogleSearch
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Web search agent
web_search_agent = Agent(
    name="web_search_agent",
    role="web researcher",
    model=Groq(id="Llama3-groq-70b-8192-tool-use-preview"),
    tools=[GoogleSearch()],
    instructions=["Include source URLs and dates in search results"],
    show_tool_calls=True,
    markdown=True
)
# Financial agent
financial_agent = Agent(
    name="finance_agent",
    role="financial analyst",
    model=Groq(id="Llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True
    )],
    instructions=["Present financial data in formatted tables"],
    show_tool_calls=False,
    markdown=True,
    debug_mode=False
)


# Multi-agent system
multi_agent = Agent(
    team=[web_search_agent, financial_agent],
    instructions=[
        "Coordinate between web search and financial analysis",
        "Use web_search_agent for general information",
        "Use finance_agent for market data and analysis"
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=False
    
)

response = multi_agent.print_response(
    "What is the current price of Apple stock?", 
    stream=True,
    
)

print(response)