import os
import javalang
import networkx as nx
import numpy as np

class JavaStructuralExtractor:
    """
    A structural dependency extraction module for RLSFLoc.
    
    Parses Java source code files using javalang to construct a directed weighted
    dependency graph using networkx. Supports nodes at the File, Method, and Statement levels,
    and extracts import, inheritance, method call, statement control flow, and statement
    data flow dependencies.
    """
    
    def __init__(self):
        # Symbol table mapping:
        # - class_name -> file_path
        # - class_name -> set of method_names
        # - "class_name.method_name" -> method_node_id
        self.class_to_file = {}
        self.class_methods = {}
        self.class_super = {}  # class_name -> superclass/interfaces
        self.symbol_table_built = False

    def build_symbol_table(self, source_files):
        """
        Builds a symbol table across all source files to resolve class, inheritance, and call targets.
        
        Parameters:
        -----------
        source_files : dict
            A dictionary mapping relative file paths to their string source code content.
        """
        for filepath, content in source_files.items():
            try:
                tree = javalang.parse.parse(content)
                for _, node in tree.filter(javalang.tree.ClassDeclaration):
                    class_name = node.name
                    self.class_to_file[class_name] = filepath
                    self.class_super[class_name] = []
                    
                    if node.extends:
                        self.class_super[class_name].append(node.extends.name)
                    if node.implements:
                        for impl in node.implements:
                            self.class_super[class_name].append(impl.name)
                            
                    if class_name not in self.class_methods:
                        self.class_methods[class_name] = set()
                    for method in node.methods:
                        self.class_methods[class_name].add(method.name)
                        
                for _, node in tree.filter(javalang.tree.InterfaceDeclaration):
                    interface_name = node.name
                    self.class_to_file[interface_name] = filepath
                    if interface_name not in self.class_methods:
                        self.class_methods[interface_name] = set()
                    for method in node.methods:
                        self.class_methods[interface_name].add(method.name)
            except Exception:
                # Silently skip unparseable or incomplete mock files
                continue
        self.symbol_table_built = True

    def extract_dependencies(self, source_files, base_weight=1.0):
        """
        Extracts structural dependencies and builds the unified directed graph.
        
        Parameters:
        -----------
        source_files : dict
            A dictionary mapping relative file paths to their string source code content.
        base_weight : float, optional (default=1.0)
            Initial default weight assigned to extracted edges.
            
        Returns:
        --------
        networkx.DiGraph
            Directed structural dependency graph with node and edge attributes.
        """
        G = nx.DiGraph()
        
        if not self.symbol_table_built:
            self.build_symbol_table(source_files)
            
        # First Pass: Create nodes and build containment hierarchy
        # Nodes: Files, Methods, Statements
        for filepath, content in source_files.items():
            file_node = f"file:{filepath}"
            G.add_node(file_node, type="file", filepath=filepath)
            
            try:
                tree = javalang.parse.parse(content)
            except Exception:
                continue  # Skip unparseable files
                
            # Import Dependencies
            for imp in tree.imports:
                imported_path = imp.path
                imported_class = imported_path.split('.')[-1]
                G.nodes[file_node]['imports'] = G.nodes[file_node].get('imports', []) + [imported_class]
                
            # Parse Classes/Interfaces
            for _, class_node in tree.filter(javalang.tree.ClassDeclaration):
                class_name = class_node.name
                
                # Inheritance Edge
                if class_name in self.class_super:
                    for super_class in self.class_super[class_name]:
                        if super_class in self.class_to_file:
                            target_file = f"file:{self.class_to_file[super_class]}"
                            G.add_edge(file_node, target_file, type="inheritance", weight=base_weight)
                            
                # Methods within Class
                for method in class_node.methods:
                    method_name = method.name
                    method_node = f"method:{filepath}:{class_name}.{method_name}"
                    G.add_node(
                        method_node,
                        type="method",
                        filepath=filepath,
                        class_name=class_name,
                        method_name=method_name
                    )
                    # Hierarchical containment edge
                    G.add_edge(file_node, method_node, type="contains", weight=0.0)
                    
                    # Control Flow & Statements within Method
                    if method.body:
                        self._build_cfg_and_statements(
                            method.body,
                            filepath,
                            class_name,
                            method_name,
                            method_node,
                            G,
                            base_weight
                        )
                        
        # Second Pass: Resolve Method Call Graphs
        # We scan all method nodes and resolve their method invocation targets
        for method_node, node_data in list(G.nodes(data=True)):
            if node_data.get('type') != 'method':
                continue
                
            filepath = node_data['filepath']
            class_name = node_data['class_name']
            method_name = node_data['method_name']
            
            # Find the AST method node
            try:
                content = source_files[filepath]
                tree = javalang.parse.parse(content)
                target_ast_method = None
                for _, c_decl in tree.filter(javalang.tree.ClassDeclaration):
                    if c_decl.name == class_name:
                        for m_decl in c_decl.methods:
                            if m_decl.name == method_name:
                                target_ast_method = m_decl
                                break
                
                if target_ast_method and target_ast_method.body:
                    for _, call_node in target_ast_method.filter(javalang.tree.MethodInvocation):
                        callee_name = call_node.member
                        qualifier = call_node.qualifier
                        
                        resolved_class = None
                        if not qualifier:
                            # Local call in the same class
                            resolved_class = class_name
                        else:
                            # Qualifier class name or object reference. Heuristic resolution:
                            # If qualifier matches a known class, resolve it.
                            if qualifier in self.class_to_file:
                                resolved_class = qualifier
                            else:
                                # Look in imports of the file
                                file_node_id = f"file:{filepath}"
                                imports = G.nodes[file_node_id].get('imports', [])
                                for imp in imports:
                                    if imp == qualifier:
                                        resolved_class = imp
                                        break
                                        
                        # Fallback heuristic: search for any class defining this method
                        if not resolved_class:
                            for c_name, methods in self.class_methods.items():
                                if callee_name in methods:
                                    resolved_class = c_name
                                    break
                                    
                        if resolved_class and callee_name in self.class_methods.get(resolved_class, set()):
                            callee_file = self.class_to_file[resolved_class]
                            target_method_node = f"method:{callee_file}:{resolved_class}.{callee_name}"
                            if G.has_node(target_method_node):
                                G.add_edge(method_node, target_method_node, type="call", weight=base_weight)
            except Exception:
                continue
                
        return G

    def _build_cfg_and_statements(self, block_statements, filepath, class_name, method_name, method_node, G, base_weight):
        """
        Builds control flow (CFG) and statement nodes inside a method block recursively,
        followed by statement-level variable data flow (def-use chains) analysis.
        """
        # Recursive statement collector
        def build_cfg_for_block(statements):
            if not statements:
                return [], []
                
            entry_nodes = []
            previous_exits = []
            
            for i, stmt in enumerate(statements):
                stmt_entries, stmt_exits = build_cfg_for_statement(stmt)
                
                if i == 0:
                    entry_nodes = stmt_entries
                    
                # Link preceding exits to current entries
                for prev in previous_exits:
                    for curr in stmt_entries:
                        G.add_edge(prev, curr, type="control_flow", weight=base_weight)
                        
                previous_exits = stmt_exits
                
            return entry_nodes, previous_exits

        def build_cfg_for_statement(stmt):
            if stmt is None:
                return [], []
                
            line = getattr(stmt, 'position', None).line if getattr(stmt, 'position', None) else 1
            node_type = type(stmt).__name__
            stmt_node = f"statement:{filepath}:{class_name}.{method_name}:{line}:{node_type}"
            
            # Create statement node
            G.add_node(
                stmt_node,
                type="statement",
                filepath=filepath,
                class_name=class_name,
                method_name=method_name,
                line=line,
                ast_type=node_type
            )
            G.add_edge(method_node, stmt_node, type="contains", weight=0.0)
            
            # Extract variables defined and used in this statement
            defs, uses = self._get_defs_and_uses(stmt)
            G.nodes[stmt_node]['defs'] = defs
            G.nodes[stmt_node]['uses'] = uses
            
            if isinstance(stmt, javalang.tree.BlockStatement):
                if stmt.statements:
                    return build_cfg_for_block(stmt.statements)
                return [stmt_node], [stmt_node]
                
            elif isinstance(stmt, javalang.tree.IfStatement):
                # Branch entries
                then_entries, then_exits = build_cfg_for_statement(stmt.then_statement)
                else_entries, else_exits = build_cfg_for_statement(stmt.else_statement)
                
                # Flow from condition to branch entries
                for entry in then_entries:
                    G.add_edge(stmt_node, entry, type="control_flow", weight=base_weight)
                for entry in else_entries:
                    G.add_edge(stmt_node, entry, type="control_flow", weight=base_weight)
                    
                # Exit nodes
                exits = then_exits + else_exits
                if not stmt.else_statement:
                    exits.append(stmt_node)  # control flows straight if cond false and no else
                    
                return [stmt_node], exits
                
            elif isinstance(stmt, (javalang.tree.WhileStatement, javalang.tree.ForStatement)):
                body_entries, body_exits = build_cfg_for_statement(stmt.body)
                
                # Flow from header to body
                for entry in body_entries:
                    G.add_edge(stmt_node, entry, type="control_flow", weight=base_weight)
                # Loop back body exits to loop header
                for body_exit in body_exits:
                    G.add_edge(body_exit, stmt_node, type="control_flow", weight=base_weight)
                    
                # Exit point is loop condition fail
                return [stmt_node], [stmt_node]
                
            elif isinstance(stmt, javalang.tree.ReturnStatement):
                # Return immediately exits method, flows to nowhere else
                return [stmt_node], []
                
            # Default simple statement
            return [stmt_node], [stmt_node]

        # 1. Build CFG
        build_cfg_for_block(block_statements)
        
        # 2. Extract Data Flow (Def-Use Chains)
        # Fetch all statement nodes in this method
        method_stmts = [
            node for node, d in G.nodes(data=True)
            if d.get('type') == 'statement' and d.get('method_name') == method_name and d.get('filepath') == filepath
        ]
        
        # Perform Reaching Definitions Data Flow Analysis per variable
        for stmt_def in method_stmts:
            defs = G.nodes[stmt_def].get('defs', [])
            for var in defs:
                # Path traversal in CFG to find reachability of this definition
                visited = set()
                queue = []
                
                # Initial successors
                for _, successor, data in G.out_edges(stmt_def, data=True):
                    if data.get('type') == 'control_flow':
                        queue.append(successor)
                        
                while queue:
                    curr = queue.pop(0)
                    if curr in visited:
                        continue
                    visited.add(curr)
                    
                    # If this statement uses the variable, add data flow edge
                    uses = G.nodes[curr].get('uses', [])
                    if var in uses:
                        G.add_edge(stmt_def, curr, type="data_flow", weight=base_weight)
                        
                    # If this statement redefines the variable, kill propagation along this path
                    redefs = G.nodes[curr].get('defs', [])
                    if var in redefs:
                        continue
                        
                    # Else continue path traversal to successors
                    for _, successor, data in G.out_edges(curr, data=True):
                        if data.get('type') == 'control_flow':
                            queue.append(successor)

    def _get_defs_and_uses(self, stmt):
        """
        Extracts local variable names defined (defs) and used (uses) by an AST statement.
        """
        defs = []
        uses = []
        
        if stmt is None:
            return defs, uses
            
        try:
            # 1. LocalVariableDeclaration
            if isinstance(stmt, javalang.tree.LocalVariableDeclaration):
                for declarator in stmt.declarators:
                    defs.append(declarator.name)
                    # Initializer uses variables
                    if declarator.initializer:
                        for _, ref in declarator.initializer.filter(javalang.tree.MemberReference):
                            uses.append(ref.member)
                            
            # 2. Assignment statement (wrapped in StatementExpression)
            elif isinstance(stmt, javalang.tree.StatementExpression) and isinstance(stmt.expression, javalang.tree.Assignment):
                assign = stmt.expression
                # Left side (def)
                if isinstance(assign.expressionl, javalang.tree.MemberReference):
                    defs.append(assign.expressionl.member)
                # Right side (use)
                for _, ref in assign.expressionr.filter(javalang.tree.MemberReference):
                    uses.append(ref.member)
                    
            # 3. Simple usage traversal for other expressions
            else:
                for _, ref in stmt.filter(javalang.tree.MemberReference):
                    uses.append(ref.member)
        except Exception:
            pass
            
        # Clean duplicates
        return list(set(defs)), list(set(uses))

    def normalize_graph_weights(self, G):
        """
        Normalizes the outgoing weights of each node in G so that they sum to 1.
        Equation 423 of the RLSFLoc article: sum_{v_j in N(v_i)} w_ij = 1.
        
        Out-of-context edge categories (like containment hierarchy 'contains')
        are excluded from the normalization summation.
        """
        for node in G.nodes():
            # Filter outgoing edges excluding containment hierarchy
            out_edges = [
                (u, v, d) for u, v, d in G.out_edges(node, data=True)
                if d.get('type') != 'contains'
            ]
            
            total_weight = sum(d.get('weight', 1.0) for u, v, d in out_edges)
            if total_weight > 0:
                for u, v, d in out_edges:
                    G[u][v]['weight'] = d.get('weight', 1.0) / total_weight
        return G
