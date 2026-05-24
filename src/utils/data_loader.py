import pandas as pd
import ast

def load_defects4j_ranking_data(filepath):
    """
    Load processed Defects4J ranking dataset.
    """
    df = pd.read_csv(filepath)
    processed = []
    for _, row in df.iterrows():
        processed.append({
            'qid': row['qid'],
            'pos_docs': ast.literal_eval(row['pos-docids']),
            'neg_docs': ast.literal_eval(row['neg-docids']),
            'type': row['type']
        })
    return processed

def load_aeeem_metrics(filepath):
    """
    Load cleaned AEEEM metric files.
    Returns: Features (DataFrame), Target Labels (Series: 1=buggy, 0=clean)
    """
    df = pd.read_csv(filepath)
    features = df.drop(columns=['id', 'class'], errors='ignore')
    target = df['class'].apply(lambda x: 1 if x == 'buggy' else 0)
    return features, target
