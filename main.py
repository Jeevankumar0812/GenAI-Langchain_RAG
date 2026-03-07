#from langchain_ollama import ChatOllama
#from langchain_openai import ChatOpenAI
from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
import os
#from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

#tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class Source(BaseModel):
    """Schema for a source used by the agent."""
    name: str = Field(..., description="The name of the source.")
    url: str = Field(..., description="The URL of the source.")

class AgentResponse(BaseModel):
    """Schema for the agent's response with answer and sources."""
    answer: str = Field(..., description="The agent's answer to the query.")
    sources: List[Source] = Field(default_factory=list, description="A list of sources used by the agent to arrive at the answer.")


@tool
def search(query: str) -> str:
    """Tool that searches over the internet.
    Args:
        query: The search query.
    Returns:
        The search results."""
    print(f"Searching for {query}...")
    #return "Tokyo is weather is sunny."
    return tavily.search(query=query)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
tools = [TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))]
agent = create_agent(model=llm, tools=tools,response_format=AgentResponse)

def main():
    print("Hello from genai-langchain-rag!")
    result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
    print(result)


if __name__ == "__main__":
    main()
