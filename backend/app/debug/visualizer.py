"""
RAG Graph Visualizer
====================

Interactive visualization of your knowledge graph using PyVis.

Features:
- Full graph visualization
- Search path visualization (see what search_context_mesh returns)
- Cluster analysis
- Export to HTML for sharing

Usage:
    from app.debug import RAGVisualizer
    from app.core import RAGStore
    
    rag = RAGStore(namespace="video_styles")
    viz = RAGVisualizer(rag)
    
    # Full graph
    viz.visualize("graph.html")
    
    # Search path for a query
    viz.visualize_search_path("energetic hook style", "search_results.html")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.rag_store import RAGStore

logger = logging.getLogger(__name__)


@dataclass
class VisualizerConfig:
    """Configuration for graph visualization."""
    
    height: str = "800px"
    width: str = "100%"
    bgcolor: str = "#ffffff"
    font_color: str = "#333333"
    
    # Node colors by type
    node_colors: dict = field(default_factory=lambda: {
        "default": "#97C2FC",
        "seed": "#FF6B6B",
        "depth_1": "#4ECDC4", 
        "depth_2": "#95E1A3",
        "depth_3": "#F7DC6F",
    })
    
    # Edge colors by relation type
    edge_colors: dict = field(default_factory=lambda: {
        "relates_to": "#888888",
        "contradicts": "#FF6B6B",
        "contains": "#4ECDC4",
        "complements": "#95E1A3",
        "requires": "#F39C12",
    })
    
    # Physics settings
    physics_enabled: bool = True
    physics_solver: str = "forceAtlas2Based"


class RAGVisualizer:
    """
    Interactive knowledge graph visualization.
    
    Generates HTML files with interactive graph visualizations
    that can be opened in any browser.
    """
    
    def __init__(self, rag: "RAGStore", config: VisualizerConfig | None = None):
        self.rag = rag
        self.client = rag.client
        self.namespace = rag.namespace
        self.config = config or VisualizerConfig()
        
        # Lazy import - only needed when visualizing
        self._Network = None
        self._nx = None
    
    def _ensure_imports(self):
        """Lazy import visualization libraries."""
        if self._Network is None:
            try:
                from pyvis.network import Network
                import networkx as nx
                self._Network = Network
                self._nx = nx
            except ImportError as e:
                raise ImportError(
                    "Visualization requires pyvis and networkx. "
                    "Install with: pip install pyvis networkx"
                ) from e
    
    def get_graph_data(self, limit: int = 200) -> tuple[list[dict], list[dict]]:
        """
        Fetch documents and relations from Supabase.
        
        Args:
            limit: Maximum number of documents to fetch
            
        Returns:
            Tuple of (documents, relations)
        """
        docs = self.client.table("documents").select(
            "id, content, metadata, created_at"
        ).eq("namespace", self.namespace).limit(limit).execute()
        
        rels = self.client.table("doc_relations").select(
            "id, source_id, target_id, type, properties"
        ).eq("namespace", self.namespace).execute()
        
        logger.info(f"Fetched {len(docs.data)} documents, {len(rels.data)} relations")
        return docs.data, rels.data
    
    def visualize(
        self,
        output_path: str | Path = "rag_graph.html",
        limit: int = 200,
        show_labels: bool = True,
    ) -> dict:
        """
        Generate interactive HTML visualization of the full knowledge graph.
        
        Args:
            output_path: Path to save the HTML file
            limit: Maximum documents to visualize
            show_labels: Whether to show text labels on nodes
            
        Returns:
            Stats about the generated visualization
        """
        self._ensure_imports()
        docs, rels = self.get_graph_data(limit)
        
        if not docs:
            logger.warning("No documents found in namespace")
            return {"nodes": 0, "edges": 0, "output": str(output_path)}
        
        # Build NetworkX graph
        G = self._nx.DiGraph()
        
        # Add nodes
        for doc in docs:
            content = doc["content"]
            label = self._truncate(content, 50) if show_labels else str(doc["id"])
            
            # Determine node color based on metadata type
            doc_type = doc.get("metadata", {}).get("type", "default")
            color = self.config.node_colors.get(doc_type, self.config.node_colors["default"])
            
            G.add_node(
                doc["id"],
                label=label,
                title=self._build_hover_text(doc),
                color=color,
                size=15,
            )
        
        # Add edges
        doc_ids = {doc["id"] for doc in docs}
        for rel in rels:
            # Only add edges where both nodes exist
            if rel["source_id"] in doc_ids and rel["target_id"] in doc_ids:
                edge_color = self.config.edge_colors.get(
                    rel["type"], 
                    self.config.edge_colors["relates_to"]
                )
                G.add_edge(
                    rel["source_id"],
                    rel["target_id"],
                    title=rel["type"],
                    color=edge_color,
                    arrows="to",
                )
        
        # Create PyVis network
        net = self._Network(
            height=self.config.height,
            width=self.config.width,
            directed=True,
            bgcolor=self.config.bgcolor,
            font_color=self.config.font_color,
        )
        
        net.from_nx(G)
        
        # Configure physics
        if self.config.physics_enabled:
            net.set_options(self._get_physics_options())
        
        net.show_buttons(filter_=["physics", "nodes", "edges"])
        
        # Save
        output_path = Path(output_path)
        net.save_graph(str(output_path))
        
        logger.info(f"Graph saved to {output_path}")
        return {
            "nodes": len(docs),
            "edges": G.number_of_edges(),
            "output": str(output_path),
        }
    
    def visualize_search_path(
        self,
        query: str,
        output_path: str | Path = "search_path.html",
        top_k: int = 10,
        graph_depth: int = 2,
    ) -> dict:
        """
        Visualize what search_context_mesh returns for a query.
        
        Shows seeds (direct matches) and graph-expanded results
        with different colors by depth.
        
        Args:
            query: Search query
            output_path: Path to save the HTML file
            top_k: Number of results
            graph_depth: Graph traversal depth
            
        Returns:
            Search results with visualization stats
        """
        self._ensure_imports()
        
        results = self.rag.search(query, top_k=top_k, graph_depth=graph_depth)
        
        if not results:
            logger.warning(f"No results for query: {query}")
            return {"query": query, "results": [], "output": str(output_path)}
        
        G = self._nx.DiGraph()
        
        # Add result nodes with color by depth
        for r in results:
            depth = r.get("depth", 0)
            source_type = r.get("source_type", "seed")
            
            if source_type == "seed":
                color = self.config.node_colors["seed"]
            else:
                color = self.config.node_colors.get(f"depth_{depth}", "#888888")
            
            # Size decreases with depth
            size = 25 - (depth * 5)
            
            label = f"[{source_type}] {self._truncate(r['content'], 40)}"
            
            G.add_node(
                r["id"],
                label=label,
                title=self._build_search_hover(r, query),
                color=color,
                size=max(size, 10),
            )
        
        # Fetch relations between returned nodes
        result_ids = [r["id"] for r in results]
        rels = self.client.table("doc_relations").select(
            "source_id, target_id, type"
        ).in_("source_id", result_ids).in_("target_id", result_ids).execute()
        
        for rel in rels.data:
            edge_color = self.config.edge_colors.get(rel["type"], "#888888")
            G.add_edge(
                rel["source_id"],
                rel["target_id"],
                title=rel["type"],
                color=edge_color,
                arrows="to",
            )
        
        # Create network
        net = self._Network(
            height="600px",
            width="100%",
            directed=True,
            bgcolor=self.config.bgcolor,
        )
        net.from_nx(G)
        net.set_options(self._get_physics_options())
        
        # Add query info as title
        net.heading = f"Search: '{query}'"
        
        output_path = Path(output_path)
        net.save_graph(str(output_path))
        
        logger.info(f"Search path visualization saved to {output_path}")
        return {
            "query": query,
            "results": results,
            "nodes_visualized": len(results),
            "edges_visualized": len(rels.data),
            "output": str(output_path),
        }
    
    def compare_searches(
        self,
        queries: list[str],
        output_path: str | Path = "search_comparison.html",
    ) -> dict:
        """
        Compare multiple search queries in one visualization.
        
        Each query's results are shown in a different color cluster.
        """
        self._ensure_imports()
        
        G = self._nx.DiGraph()
        
        # Color palette for different queries
        query_colors = ["#FF6B6B", "#4ECDC4", "#95E1A3", "#F7DC6F", "#BB8FCE"]
        
        all_results = {}
        for i, query in enumerate(queries):
            color = query_colors[i % len(query_colors)]
            results = self.rag.search(query, top_k=5, graph_depth=1)
            all_results[query] = results
            
            for r in results:
                # Don't duplicate nodes, but track which queries found them
                if r["id"] not in G:
                    G.add_node(
                        r["id"],
                        label=self._truncate(r["content"], 40),
                        title=f"Found by: {query}\n\n{r['content']}",
                        color=color,
                        size=20,
                    )
        
        net = self._Network(height="700px", width="100%", directed=True)
        net.from_nx(G)
        
        output_path = Path(output_path)
        net.save_graph(str(output_path))
        
        return {
            "queries": queries,
            "results": all_results,
            "output": str(output_path),
        }
    
    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text with ellipsis."""
        text = text.replace("\n", " ").strip()
        if len(text) > max_len:
            return text[:max_len] + "..."
        return text
    
    def _build_hover_text(self, doc: dict) -> str:
        """Build hover text for a document node."""
        lines = [
            f"ID: {doc['id']}",
            f"Content: {doc['content'][:200]}...",
        ]
        if doc.get("metadata"):
            lines.append(f"Metadata: {json.dumps(doc['metadata'], indent=2)}")
        return "\n".join(lines)
    
    def _build_search_hover(self, result: dict, query: str) -> str:
        """Build hover text for a search result node."""
        lines = [
            f"Query: {query}",
            f"Score: {result.get('score', 'N/A'):.4f}" if result.get('score') else "",
            f"Depth: {result.get('depth', 0)}",
            f"Source: {result.get('source_type', 'seed')}",
            "",
            f"Content: {result['content'][:300]}...",
        ]
        return "\n".join(filter(None, lines))
    
    def _get_physics_options(self) -> str:
        """Get PyVis physics options as JSON string."""
        return json.dumps({
            "physics": {
                "enabled": self.config.physics_enabled,
                "solver": self.config.physics_solver,
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08,
                },
                "stabilization": {
                    "iterations": 150,
                },
            }
        })
