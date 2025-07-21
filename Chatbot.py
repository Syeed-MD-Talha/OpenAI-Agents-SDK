import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, SQLiteSession, OpenAIChatCompletionsModel

# Load environment variables from .env file
load_dotenv()

async def create_gemini_model() -> OpenAIChatCompletionsModel:
    """Create and configure the Gemini model."""
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    MODEL_NAME = 'gemini-2.0-flash'
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key)
    return OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=client)


async def main():
    """Main chat application."""

    gemini_model = await create_gemini_model()

    while True:
        user_input=input("Enter your question (or type 'exit' to quit):")
        if user_input.lower() == "exit":
            break
        # Create an agent
        agent=Agent(
            name="Assistant",
            instructions="Reply very concisely.",
            model=gemini_model,
        )
        
        session = SQLiteSession("conversation_123", "conversation_history_1.db")

        result  = await Runner.run(
            agent,
            user_input,
            session=session
        )
        print(f"Assistant: {result.final_output}\n")


if __name__ == "__main__":
    asyncio.run(main())
