import numpy as np
import pandas as pd

class JavaSemanticEmbedder:
    """
    A semantic embedding and similarity module for RLSFLoc.
    
    Loads a SentenceTransformer model (defaults to all-MiniLM-L6-v2) and computes
    cosine similarity scores between preprocessed statement texts and a bug report
    or failing test description.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Parameters:
        -----------
        model_name : str, optional (default='all-MiniLM-L6-v2')
            The name of the pre-trained SentenceTransformer model to load.
        """
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        """
        Lazy-loads the SentenceTransformer model to avoid heavy startup overhead
        on import.
        """
        if self.model is None:
            # Import inside method for fast module startup
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)

    def compute_semantic_scores(self, statement_df, bug_report):
        """
        Computes semantic similarity scores between statement texts and a bug report.
        
        Mathematical Formulation (Equation 451 from research paper):
        -----------------------------------------------------------
        S_sem(v_i) = (z_i^T * z_B) / (||z_i||_2 * ||z_B||_2)
        
        Where:
          - z_i is the high-dimensional embedding vector of statement i
          - z_B is the embedding vector of the bug report description
        
        Parameters:
        -----------
        statement_df : pandas.DataFrame
            DataFrame containing 'statement_id' and 'semantic_text' columns.
        bug_report : str
            The bug report or failing test description text.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with columns: ['statement_id', 'semantic_score']
            Ordered identically to the input DataFrame rows.
        """
        # Handle empty inputs gracefully
        if statement_df.empty:
            return pd.DataFrame(columns=['statement_id', 'semantic_score'])
            
        if 'statement_id' not in statement_df.columns or 'semantic_text' not in statement_df.columns:
            raise ValueError("statement_df must contain 'statement_id' and 'semantic_text' columns")

        # 1. Lazy load model
        self._load_model()
        
        # 2. Extract and handle potentially empty/null semantic texts
        statements = statement_df['semantic_text'].fillna("").tolist()
        
        # 3. Encode statement texts in a single batch (highly optimized!)
        statement_embeddings = self.model.encode(statements, show_progress_bar=False)
        
        # 4. Encode the bug report
        bug_embedding = self.model.encode(bug_report, show_progress_bar=False)
        
        # 5. Compute vectorized cosine similarities (Equation 451)
        # statement_embeddings has shape (num_statements, dim)
        # bug_embedding has shape (dim,)
        
        # Normalization to unit length (add 1e-12 epsilon to avoid division by zero)
        stmt_norms = np.linalg.norm(statement_embeddings, axis=1, keepdims=True)
        stmt_norms = np.where(stmt_norms > 0.0, stmt_norms, 1.0)
        stmt_embeddings_normalized = statement_embeddings / stmt_norms
        
        bug_norm = np.linalg.norm(bug_embedding)
        bug_norm = bug_norm if bug_norm > 0.0 else 1.0
        bug_embedding_normalized = bug_embedding / bug_norm
        
        # Vectorized dot product calculates all cosine similarities instantly in O(N * dim)
        similarities = stmt_embeddings_normalized.dot(bug_embedding_normalized)
        
        # 6. Package output
        results_df = pd.DataFrame({
            'statement_id': statement_df['statement_id'].values,
            'semantic_score': similarities
        })
        
        return results_df
