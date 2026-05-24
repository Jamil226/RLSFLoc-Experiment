import pytest
import pandas as pd

from src.utils.semantic_preprocessor import JavaSemanticPreprocessor

def test_camel_case_splitting():
    """
    Verify camelCase and PascalCase splitting behavior.
    """
    preprocessor = JavaSemanticPreprocessor()
    
    # 1. Standard PascalCase
    res1 = preprocessor.split_camel_case("ArrayIndexOutOfBoundsException")
    assert res1 == ["Array", "Index", "Out", "Of", "Bounds", "Exception"]
    
    # 2. Standard camelCase
    res2 = preprocessor.split_camel_case("myVariableIdentifier")
    assert res2 == ["my", "Variable", "Identifier"]
    
    # 3. Camel Case with numbers
    res3 = preprocessor.split_camel_case("camel2camel")
    assert res3 == ["camel2camel"]  # Default regex leaves numbers within tokens unchanged or matches as single token


def test_token_cleaning_and_stopword_filtering():
    """
    Verify non-alphanumeric cleaning and stopword/keyword filtering.
    """
    preprocessor = JavaSemanticPreprocessor()
    
    # 1. Remove symbols and split camelCase
    res1 = preprocessor.clean_token("!#calculateSuspiciousnessScore()")
    assert res1 == ["calculate", "suspiciousness", "score"]
    
    # 2. Filter English stopwords
    res2 = preprocessor.clean_token("the")
    assert res2 == []
    
    # 3. Filter Java keywords
    res3 = preprocessor.clean_token("synchronized")
    assert res3 == []
    
    # 4. Keep meaningful identifiers
    res4 = preprocessor.clean_token("payment")
    assert res4 == ["payment"]


def test_comment_extraction():
    """
    Verify proximity comment parser extracts line and block comments correctly with correct lines.
    """
    source_code = """package com.example;
    
    // This is a line comment on line 3
    public class Sample {
        /*
         * This is a block
         * comment on lines 5 to 7
         */
        public void run() {
            int a = 1;
        }
    }
    """
    preprocessor = JavaSemanticPreprocessor()
    comments = preprocessor.extract_comments_with_proximity(source_code)
    
    assert len(comments) == 2
    # Line comment at line 3
    assert comments[0] == (3, "This is a line comment on line 3")
    # Block comment starts at line 5
    assert comments[1][0] == 5
    assert "This is a block" in comments[1][1]
    assert "comment on lines" in comments[1][1]


def test_semantic_preprocessor_end_to_end():
    """
    Test end-to-end semantic preprocessing of a mock Java file.
    
    Verify that:
      - statement_id matches structural extractor signature perfectly
      - semantic_text contains preprocessed subwords (split, lowercase, stopwords filtered)
      - Proximity comments are appended to semantic_text
    """
    java_code = """
    package com.example;
    
    public class Payment {
        public void processPayment() {
            // Line comment immediately above
            int transactionVal = 100;
            
            System.out.println("Processing successful transaction");
        }
    }
    """
    source_files = {
        "src/com/example/Payment.java": java_code
    }
    
    preprocessor = JavaSemanticPreprocessor()
    df = preprocessor.preprocess_all(source_files)
    
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['statement_id', 'semantic_text']
    assert len(df) == 2  # LocalVariableDeclaration and StatementExpression (System.out.println)
    
    # Examine first statement: int transactionVal = 100;
    stmt1 = df.iloc[0]
    assert "statement:src/com/example/Payment.java:Payment.processPayment:" in stmt1['statement_id']
    assert "LocalVariableDeclaration" in stmt1['statement_id']
    
    # Semantic text of first statement should contain:
    # - class context: 'payment'
    # - method context: 'process', 'payment' (deduplicated)
    # - declarator: 'transaction', 'val' (split camelCase)
    # - proximity comment: 'line', 'comment', 'immediately' (split, stopwords filtered)
    terms1 = stmt1['semantic_text'].split()
    assert 'payment' in terms1
    assert 'process' in terms1
    assert 'transaction' in terms1
    assert 'val' in terms1
    assert 'comment' in terms1
    
    # Exclude stopwords & keywords
    assert 'public' not in terms1
    assert 'void' not in terms1
    assert 'int' not in terms1
    assert 'is' not in terms1
    
    # Examine second statement: System.out.println(...)
    stmt2 = df.iloc[1]
    terms2 = stmt2['semantic_text'].split()
    assert 'system' in terms2
    assert 'out' not in terms2 # 'out' is an English stopword!
    assert 'println' in terms2
    assert 'processing' in terms2
    assert 'successful' in terms2
    assert 'transaction' in terms2
    
    print("\n[Verification Log] Semantic Preprocessor end-to-end tests passed.")
