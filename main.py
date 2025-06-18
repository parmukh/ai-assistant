from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool,Tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import httpx
import os

#### for local use only ########
original_init = httpx._client.Client.__init__
def no_verify_init(self, *args, **kwargs):
    kwargs["verify"] = False
    original_init(self, *args, **kwargs)

httpx._client.Client.__init__ = no_verify_init
##### for local use only end #####
load_dotenv()
@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmeric calculations with numbers"""
    print("Tool has been called.")
    return f"The sum of {a} and {b} is {a + b}"
    
@tool
def say_hello(name: str) -> str:
    """Useful for greeting a user"""
    print("Tool has been called.")
    return f"Hello {name}, I hope you are well today"

import requests

def get_mcp_data(query: str) -> str:
    try:
        response = requests.post(
            url="http://localhost:8000/mcp/query",  # replace with your MCP endpoint
            json={"query": query},
            timeout=5,
        )
        response.raise_for_status()
        return response.json().get("result", "No result found.")
    except requests.exceptions.RequestException as e:
        return f"MCP API error: {str(e)}"
def main():
    model = AzureChatOpenAI(
    azure_deployment="gpt-4.1",  # same as your deployment name
    azure_endpoint="https://abc.openai.azure.com",  # no `/openai` at the end
    api_key=os.getenv("OPENAI_API_KEY"), 
    api_version="2025-01-01-preview" # or hardcode just to test
)
    mcp_tool = Tool(
    name="MCPTool",
    func=get_mcp_data,
    description="Use this tool to query the MCP system for system status, versions, or internal data."
)

    tools = [calculator, say_hello,mcp_tool]
    agent_executor = create_react_agent(model, tools)
    
    print("\nAssistant: Welcome! I'm your AI assistant. Type 'quit' to exit.")
    print("\n You can ask me to perform calculations or chat with me.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input == "quit":
            print("\nAssistant:Goodbye!")
            break
        
        print("\nAssistant: ", end="")
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print()

if __name__ == "__main__":
    main()
