from ddgs import DDGS
from langchain_core.tools import tool
import trafilatura
from tools.preprocess import CustomDocumentLoader, split_text
from tools.data import save_embeddings
import asyncio
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_fetch_content(url: str) -> str:
    """
    Fetches the content from a given URL, extracts it with trafilatura,
    processes it with a custom document loader, splits it into chunks,
    and saves embeddings. The results are cached for repeated queries.
    """
    try:
        content = trafilatura.fetch_url(url)
        if content:
            return trafilatura.extract(content, favor_recall=True)
    except Exception:
        pass
    return None

@lru_cache(maxsize=128)
def cached_search_content(query: str, source: str) -> list:
    """
    Performs a DuckDuckGo search query for the specified source (text or news)
    and caches the raw result list.
    """
    clean_q = str(query).strip().strip("`'\"\n\r\t ")
    results = []
    try:
        with DDGS() as ddgs:
            if source == "news":
                items = list(ddgs.news(clean_q, max_results=6))
                for item in items:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("url", ""),
                        "snippet": item.get("body", "")
                    })
            else:
                items = list(ddgs.text(clean_q, max_results=6))
                for item in items:
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("href", ""),
                        "snippet": item.get("body", "")
                    })
    except Exception as e:
        print(f"ddgs search error: {e}")
    return results

@tool("Search",parse_docstring=True)
async def fetch_sites(query : str) -> str:
    """
    A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. This returns only the answer - not the original source data.

    Args:
        query (str): The search term or question to query.

    Returns:
        str: The result or answer retrieved from the web search, excluding the source.
    """
    clean_q = str(query).strip().strip("`'\"\n\r\t ")

    # Web search
    web_search_task = asyncio.create_task(asyncio.to_thread(lambda: cached_search_content(clean_q, "text")))

    # News search
    news_search_task = asyncio.create_task(asyncio.to_thread(lambda: cached_search_content(clean_q, "news")))

    web_ret, news_ret = await asyncio.gather(web_search_task, news_search_task)

    # Combine results and eliminate duplicate web results
    unique_results = {}
    for item in (news_ret + web_ret):
        if item.get('link') and item['link'] not in unique_results:
            unique_results[item['link']] = item
    
    ret = list(unique_results.values())

    fetched = []
    top_k=8
    content=""
    # Create tasks for text search results
    web_tasks = [visit.ainvoke({"query": row.get("link", "")}) for row in ret[:top_k]]
    web_results = await asyncio.gather(*web_tasks)
    
    for i, row in enumerate(ret[:top_k]):
        scraped = web_results[i] if i < len(web_results) else None
        row_content = scraped if (scraped and scraped != "No content could be extracted from the webpage.") else row.get("snippet")
        row["content"] = row_content
        if row_content:
            content = content + '\n\nTitle: ' + row.get('title', '') + '\nLink: ' + row.get('link', '')
            content = content + '\nContent: ' + str(row_content)
        fetched.append(row)

    return str(content+'\n\n')

@tool("OpenLink",parse_docstring=True)
async def visit(query: str) -> str:
    """
    Extracts webpage content from URL. Input: string URL. Returns: extracted text content

    Args:
        query (str): URL string or dictionary containing the key "link" with the URL of the webpage to extract content from.

    Returns:
        str: The extracted content from the webpage, excluding the source or metadata.
    """
    if isinstance(query, str):

        url = query.strip().strip("`'\"\n\r\t ")
        query_dict = {"link": url}
    elif isinstance(query, dict):
        url = query.get("link", "")
        query_dict = query
    else:
        url = str(query)
        query_dict = {"link": url}

    content = cached_fetch_content(url)
    query_dict["content"] = content
    if content is None:
        return "No content could be extracted from the webpage."

    # Process the document and save embeddings
    await process_and_save(query_dict)
    return content or "No content could be extracted from the webpage."


async def process_and_save(query):
    """
    Background task to process documents and save embeddings.
    """
    try:
        loader = CustomDocumentLoader(query)
        documents = []
        async for doc in loader.lazy_load():
            documents.append(doc)
        chunks = await split_text(documents, 2048, 512)
        batch_size = 10
        for i in range(0,len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            # Process the batch of documents
            await save_embeddings(batch)
    except Exception as e:
        print(f"Background processing error: {str(e)}")