from fastmcp import FastMCP

# from tools.fetch.fetch_full_paper import fetch_full_paper
# from tools.fetch.fetch_paper_artifacts import fetch_paper_artifacts
# from tools.metadata.get_author_top_works import get_author_top_works
# from tools.metadata.query_paper_metadata import query_paper_metadata
from tools import find_relevant_papers, get_citations, get_references

# Initialize the server
mcp = FastMCP("Paper Weight Research Assistant")

# Register all tools
# mcp.add_tool(fetch_full_paper)
# mcp.add_tool(fetch_paper_artifacts)
# mcp.add_tool(get_author_top_works)
mcp.add_tool(get_citations)
mcp.add_tool(get_references)
mcp.add_tool(find_relevant_papers)

if __name__ == "__main__":
    mcp.run()