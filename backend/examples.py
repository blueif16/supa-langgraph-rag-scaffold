"""
Examples - How to use the RAG scaffold
======================================
"""
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Example 1: Basic usage
def basic():
    from app.core import RAGStore
    
    rag = RAGStore(namespace="demo")
    
    # Ingest rich, self-describing content
    rag.ingest("""
    Product Hunt hook pacing: Fast energetic openings, 1.5-2 second cuts in first 
    10 seconds. High energy, immediate value prop, no slow builds. Cold open 
    straight into product. Works for SaaS, dev tools, productivity apps.
    """, source="ph_analysis_2025", type="pattern")
    
    # Search by feeling
    results = rag.search("fast energetic opening style")
    for r in results:
        print(r["content"][:200])


# Example 2: Batch from data
def batch():
    from app.core import RAGStore, DataAdapter
    
    rag = RAGStore(namespace="styles")
    
    # From any source
    contents = DataAdapter.from_json_file("patterns.json", content_field="description")
    rag.ingest_batch(contents, source="patterns_file")


# Example 3: LangGraph agent
def agent():
    from app.core import RAGStore, create_search_tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    
    rag = RAGStore(namespace="agent_kb")
    tool = create_search_tool(rag, name="search", description="Search knowledge")
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    agent = create_react_agent(llm, tools=[tool])
    
    result = agent.invoke({"messages": [{"role": "user", "content": "What's a good hook style?"}]})
    print(result["messages"][-1].content)


# Example 4: Custom workflow
def custom_workflow():
    from app.core import RAGStore
    from app.graph.workflow import create_workflow
    
    rag = RAGStore(namespace="custom")
    wf = create_workflow(rag, max_retries=2)
    
    result = wf.invoke({"question": "How to make energetic intros?"})
    print(result["messages"][-1].content)


# Example 5: Graph relations
def graph():
    from app.core import RAGStore
    
    rag = RAGStore(namespace="connected")
    
    r1 = rag.ingest("Fast cuts create energy and urgency.")
    r2 = rag.ingest("Slow pacing allows comprehension of complex features.")
    r3 = rag.ingest("Hook should be fast, demo can be slower.")
    
    rag.add_relation(r3["id"], r1["id"], "references")
    rag.add_relation(r3["id"], r2["id"], "references")
    
    # Search finds r3, graph expands to r1 and r2
    results = rag.search("pacing advice")
    print(f"Found {len(results)} results (including graph-connected)")


if __name__ == "__main__":
    print("Run individual functions to test")
