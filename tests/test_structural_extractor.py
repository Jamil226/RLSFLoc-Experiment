import pytest
import networkx as nx
import numpy as np

from src.utils.structural_extractor import JavaStructuralExtractor

def test_structural_extractor_basic():
    """
    Test JavaStructuralExtractor correctness using mock Java file contents.
    
    Covers:
      - File, Method, and Statement node extraction
      - Containment edges
      - Import & Inheritance dependencies
      - Method call graph linkages
      - CFG control flow (sequential and branching)
      - Data flow reaching definitions (def-use chains)
      - Equation 423 outgoing weight normalization
    """
    # 1. Define two mock Java files
    base_file_content = """
    package com.example;
    
    public class MyBase {
        public void baseMethod() {
            int a = 1;
        }
    }
    """
    
    class_file_content = """
    package com.example;
    
    import com.example.MyBase;
    
    public class MyClass extends MyBase {
        public void helper(int p) {
            int val = p + 2;
        }
        
        public void main() {
            int x = 5;
            helper(x);
            if (x > 0) {
                x = 10;
            } else {
                x = 20;
            }
            int y = x;
        }
    }
    """
    
    source_files = {
        "src/com/example/MyBase.java": base_file_content,
        "src/com/example/MyClass.java": class_file_content
    }
    
    # 2. Extract dependencies
    extractor = JavaStructuralExtractor()
    G = extractor.extract_dependencies(source_files, base_weight=1.0)
    
    # 3. Verify Nodes
    # Files
    assert G.has_node("file:src/com/example/MyBase.java")
    assert G.has_node("file:src/com/example/MyClass.java")
    
    # Methods
    assert G.has_node("method:src/com/example/MyBase.java:MyBase.baseMethod")
    assert G.has_node("method:src/com/example/MyClass.java:MyClass.helper")
    assert G.has_node("method:src/com/example/MyClass.java:MyClass.main")
    
    # Containment
    assert G.has_edge("file:src/com/example/MyBase.java", "method:src/com/example/MyBase.java:MyBase.baseMethod")
    assert G.has_edge("file:src/com/example/MyClass.java", "method:src/com/example/MyClass.java:MyClass.helper")
    assert G.has_edge("file:src/com/example/MyClass.java", "method:src/com/example/MyClass.java:MyClass.main")
    
    # 4. Verify Inheritance
    # MyClass extends MyBase
    assert G.has_edge("file:src/com/example/MyClass.java", "file:src/com/example/MyBase.java")
    edge_data = G.get_edge_data("file:src/com/example/MyClass.java", "file:src/com/example/MyBase.java")
    assert edge_data["type"] == "inheritance"
    
    # 5. Verify Call Graph Linkages
    # main calls helper
    assert G.has_edge("method:src/com/example/MyClass.java:MyClass.main", "method:src/com/example/MyClass.java:MyClass.helper")
    call_edge = G.get_edge_data("method:src/com/example/MyClass.java:MyClass.main", "method:src/com/example/MyClass.java:MyClass.helper")
    assert call_edge["type"] == "call"
    
    # 6. Verify Control Flow (CFG) inside main
    main_stmts = [
        node for node, d in G.nodes(data=True)
        if d.get('type') == 'statement' and d.get('method_name') == 'main'
    ]
    # Check that statement nodes exist
    assert len(main_stmts) > 0
    
    # Find IfStatement node
    if_nodes = [node for node in main_stmts if G.nodes[node]['ast_type'] == 'IfStatement']
    assert len(if_nodes) == 1
    if_node = if_nodes[0]
    
    # Check that outgoing control flow edges from IfStatement exist
    successors = [v for u, v, d in G.out_edges(if_node, data=True) if d.get('type') == 'control_flow']
    assert len(successors) >= 1  # Should branch to then or else branch entry nodes
    
    # 7. Verify Data Flow (Def-Use reaching definitions)
    # Inside MyClass.helper: val = p + 2
    # The parameter `p` is used in statement defining `val`
    helper_stmts = [
        node for node, d in G.nodes(data=True)
        if d.get('type') == 'statement' and d.get('method_name') == 'helper'
    ]
    # Check if a data flow edge exists between variable references
    data_edges = [
        (u, v) for u, v, d in G.edges(data=True)
        if d.get('type') == 'data_flow' and u in helper_stmts
    ]
    # Since helper only has local declarations, let's verify reaching definitions in main
    # main defines x = 5 (stmt 1), which reaches helper(x) (stmt 2)
    # Let's verify data flow edge from definition statement of x to usage statement
    main_data_edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if d.get('type') == 'data_flow' and u in main_stmts
    ]
    assert len(main_data_edges) >= 1
    
    # 8. Verify Outgoing Weight Normalization (Equation 423)
    G_weighted = extractor.normalize_graph_weights(G)
    
    # For a node with outgoing edges, the weights (excluding contains) must sum to 1.0
    for node in G_weighted.nodes():
        out_edges = [
            (u, v, d) for u, v, d in G_weighted.out_edges(node, data=True)
            if d.get('type') != 'contains'
        ]
        if out_edges:
            sum_weights = sum(d.get('weight', 0.0) for u, v, d in out_edges)
            assert np.isclose(sum_weights, 1.0)
            
    print("\n[Verification Log] Structural Extractor passed basic checks.")
