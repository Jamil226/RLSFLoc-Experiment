import os
import re
import javalang
import pandas as pd

class JavaSemanticPreprocessor:
    """
    A semantic preprocessing module for RLSFLoc.
    
    Extracts class names, method names, variable names, identifiers, external API calls,
    and comments at the statement level from Java source code files. Performs CamelCase
    splitting, lowercase normalization, and Java/English stopword filtering to preserve
    meaningful program semantics.
    """
    
    def __init__(self):
        # English stopwords
        self.stopwords_en = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
            "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
            "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
            "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
            "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
            "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
            "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
            "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
            "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
            "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
            "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
            "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
            "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
            "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
            "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
            "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
            "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
        }
        
        # Java keywords
        self.keywords_java = {
            "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
            "class", "const", "continue", "default", "do", "double", "else", "enum",
            "extends", "final", "finally", "float", "for", "goto", "if", "implements",
            "import", "instanceof", "int", "interface", "long", "native", "new", "package",
            "private", "protected", "public", "return", "short", "static", "strictfp",
            "super", "switch", "synchronized", "this", "throw", "throws", "transient",
            "try", "void", "volatile", "while", "true", "false", "null"
        }

    def split_camel_case(self, token):
        """
        Splits camelCase or PascalCase identifiers into a list of words.
        Example: "ArrayIndexOutOfBounds" -> ["Array", "Index", "Out", "Of", "Bounds"]
        """
        s = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token)
        s = re.sub('([A-Z])([A-Z][a-z])', r'\1 \2', s)
        return s.split()

    def clean_token(self, token):
        """
        Applies cleaning, camelCase splitting, lowercasing, and stopword filtering.
        Preserves symbol-delimited boundaries (such as dot package/method accesses)
        as separate semantic words.
        """
        if not token:
            return []
            
        # Replace non-alphanumeric symbols with spaces to preserve word boundaries
        token_space = re.sub(r'[^a-zA-Z0-9]', ' ', token)
        words = token_space.split()
        
        cleaned = []
        for word in words:
            subwords = self.split_camel_case(word)
            for w in subwords:
                w_lower = w.lower()
                if w_lower not in self.stopwords_en and w_lower not in self.keywords_java and len(w_lower) > 1:
                    cleaned.append(w_lower)
        return cleaned

    def extract_comments_with_proximity(self, source_code):
        """
        Extracts comments (both line and block) from raw Java source code
        along with their corresponding line numbers.
        """
        comments = []
        lines = source_code.splitlines()
        
        in_block_comment = False
        block_comment_buffer = []
        block_start_line = 1
        
        for idx, line in enumerate(lines):
            line_num = idx + 1
            line_stripped = line.strip()
            
            # 1. Handle block comments
            if in_block_comment:
                if '*/' in line_stripped:
                    end_idx = line_stripped.find('*/')
                    block_comment_buffer.append(line_stripped[:end_idx])
                    comment_text = " ".join(block_comment_buffer)
                    # Clean out non-alphanumeric chars
                    comment_text = re.sub(r'[/*\s]+', ' ', comment_text).strip()
                    comments.append((block_start_line, comment_text))
                    in_block_comment = False
                    block_comment_buffer = []
                else:
                    block_comment_buffer.append(line_stripped)
                continue
                
            if line_stripped.startswith('/*'):
                in_block_comment = True
                block_start_line = line_num
                if '*/' in line_stripped:  # single line block comment
                    end_idx = line_stripped.find('*/')
                    comment_text = line_stripped[2:end_idx].strip()
                    comments.append((line_num, comment_text))
                    in_block_comment = False
                else:
                    block_comment_buffer.append(line_stripped[2:])
                continue
                
            # 2. Handle line comments
            if '//' in line_stripped:
                start_idx = line_stripped.find('//')
                comment_text = line_stripped[start_idx + 2:].strip()
                comments.append((line_num, comment_text))
                
        return comments

    def preprocess_source(self, filepath, content):
        """
        Parses a single Java file and extracts clean semantic texts for each statement.
        
        Returns:
        --------
        list of dict
            List containing dictionaries with 'statement_id' and 'semantic_text' fields.
        """
        processed_statements = []
        
        try:
            tree = javalang.parse.parse(content)
        except Exception:
            return []  # Skip unparseable files
            
        # Extract comments
        comments = self.extract_comments_with_proximity(content)
        
        # We iterate over classes, methods, and statements
        for _, class_node in tree.filter(javalang.tree.ClassDeclaration):
            class_name = class_node.name
            class_tokens = self.clean_token(class_name)
            
            for method in class_node.methods:
                method_name = method.name
                method_tokens = self.clean_token(method_name)
                
                if not method.body:
                    continue
                    
                # We walk all statements inside the method
                # (Same logic as in structural_extractor to match statement_ids perfectly)
                statements_in_body = []
                def collect_statements(stmt):
                    if stmt is None:
                        return
                    statements_in_body.append(stmt)
                    
                    if isinstance(stmt, javalang.tree.BlockStatement):
                        if stmt.statements:
                            for s in stmt.statements:
                                collect_statements(s)
                    elif isinstance(stmt, javalang.tree.IfStatement):
                        collect_statements(stmt.then_statement)
                        collect_statements(stmt.else_statement)
                    elif isinstance(stmt, (javalang.tree.WhileStatement, javalang.tree.ForStatement)):
                        collect_statements(stmt.body)
                        
                for stmt in method.body:
                    collect_statements(stmt)
                    
                for stmt in statements_in_body:
                    line = getattr(stmt, 'position', None).line if getattr(stmt, 'position', None) else 1
                    node_type = type(stmt).__name__
                    
                    # Reconstruction of statement_id
                    stmt_id = f"statement:{filepath}:{class_name}.{method_name}:{line}:{node_type}"
                    
                    stmt_semantic_terms = []
                    
                    # A. Add Class and Method context words
                    stmt_semantic_terms.extend(class_tokens)
                    stmt_semantic_terms.extend(method_tokens)
                    
                    # B. Extract variable names and identifiers from AST node
                    try:
                        # Declarations
                        for _, decl in stmt.filter(javalang.tree.VariableDeclarator):
                            stmt_semantic_terms.extend(self.clean_token(decl.name))
                        # References
                        for _, ref in stmt.filter(javalang.tree.MemberReference):
                            stmt_semantic_terms.extend(self.clean_token(ref.member))
                        # API and method calls
                        for _, call in stmt.filter(javalang.tree.MethodInvocation):
                            stmt_semantic_terms.extend(self.clean_token(call.member))
                            if call.qualifier:
                                stmt_semantic_terms.extend(self.clean_token(call.qualifier))
                        # Literals
                        for _, lit in stmt.filter(javalang.tree.Literal):
                            if isinstance(lit.value, str):
                                # Clean string quotes and split
                                clean_val = lit.value.replace('"', '').replace("'", "")
                                for word in clean_val.split():
                                    stmt_semantic_terms.extend(self.clean_token(word))
                    except Exception:
                        pass
                        
                    # C. Proximity comment association (Equation proximity)
                    # Associate comments within 2 lines above the statement
                    for c_line, c_text in comments:
                        if 0 <= (line - c_line) <= 2:
                            for word in c_text.split():
                                stmt_semantic_terms.extend(self.clean_token(word))
                                
                    # Clean out duplicates while keeping order
                    unique_terms = []
                    for term in stmt_semantic_terms:
                        if term not in unique_terms:
                            unique_terms.append(term)
                            
                    semantic_str = " ".join(unique_terms)
                    
                    processed_statements.append({
                        'statement_id': stmt_id,
                        'semantic_text': semantic_str
                    })
                    
        return processed_statements

    def preprocess_all(self, source_files):
        """
        Preprocesses all source files and packages the output into a single Pandas DataFrame.
        
        Parameters:
        -----------
        source_files : dict
            A dictionary mapping relative file paths to their string source code content.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with columns: ['statement_id', 'semantic_text']
        """
        all_records = []
        for filepath, content in source_files.items():
            records = self.preprocess_source(filepath, content)
            all_records.extend(records)
            
        return pd.DataFrame(all_records, columns=['statement_id', 'semantic_text'])
