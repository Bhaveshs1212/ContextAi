from dotenv import find_dotenv, load_dotenv
import os
import asyncio
import logging
from langchain.agents import create_agent
from langchain_core.caches import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from tools.web_scraper import fetch_sites, visit
from tools.relevance_search import fetch_information

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(find_dotenv(usecwd=True), override=True)

# Retrieve model identifier from env variables
MODEL = os.environ.get("MODEL", "google_genai:gemini-3.5-flash-lite")

# Define tools for web search and content extraction
tools = [fetch_sites, visit, fetch_information]

# Create agent
rag_agent = create_agent(
    model=MODEL,
    tools=tools,
    system_prompt="You are LinkMind, an intelligent AI research assistant developed by Daniyal Anis. Always provide clear answers and cite sources whenever available.",
    checkpointer=InMemorySaver(),
    cache=InMemoryCache(),
)

