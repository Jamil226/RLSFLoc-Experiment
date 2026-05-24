import pytest
import pandas as pd
import numpy as np

from src.utils.semantic_embedder import JavaSemanticEmbedder

def test_semantic_embedding_cosine_similarity():
    """
    Test semantic similarity scoring using JavaSemanticEmbedder with the all-MiniLM-L6-v2 model.
    
    Verifies:
      - DataFrame output format (statement_id, semantic_score)
      - Cosine similarity values are mathematically bounded in [-1.0, 1.0]
      - Statements matching the semantic domain of the bug report receive higher scores.
    """
    # 1. Prepare three mock statements
    statement_df = pd.DataFrame([
        {
            'statement_id': 'statement:src/Payment.java:Payment.process:12:LocalVariableDeclaration',
            'semantic_text': 'payment processing transaction value unsuccessful'
        },
        {
            'statement_id': 'statement:src/Database.java:Database.query:24:StatementExpression',
            'semantic_text': 'query database table select user profile record'
        },
        {
            'statement_id': 'statement:src/Config.java:Config.load:8:IfStatement',
            'semantic_text': 'load configuration file path properties custom property'
        }
    ])
    
    # 2. Bug report describes payment processing failure
    bug_report = "unsuccessful credit card payment gateway processing transaction"
    
    # 3. Instantiate and compute scores
    embedder = JavaSemanticEmbedder(model_name='all-MiniLM-L6-v2')
    res = embedder.compute_semantic_scores(statement_df, bug_report)
    
    # 4. Asserts
    assert isinstance(res, pd.DataFrame)
    assert list(res.columns) == ['statement_id', 'semantic_score']
    assert len(res) == 3
    
    scores = res['semantic_score'].values
    
    # Bounded in [-1.0, 1.0]
    assert (scores >= -1.0).all() and (scores <= 1.0).all()
    
    # The statement describing payment processing (index 0) must score significantly higher
    # than the other two statements because of semantic relevance.
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    
    # Statement 0 should have high similarity (typically > 0.5) while others should have low
    assert scores[0] > 0.5
    
    print(f"\n[Verification Log] Semantic Embedder passed cosine similarity checks.")
    print(f"  - Payment Statement score: {scores[0]:.4f}")
    print(f"  - Database Statement score: {scores[1]:.4f}")
    print(f"  - Config Statement score: {scores[2]:.4f}")
