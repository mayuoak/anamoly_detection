from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import pdb
import ast
import json
from sklearn.preprocessing import OneHotEncoder

results = {}
def process(contam, df: pd.DataFrame):
    model = IsolationForest(contamination=contam, random_state=99)
    X_embeddings = np.vstack(df['embedding'].apply(ast.literal_eval).values)
    sentiment_encoder = OneHotEncoder()
    sentiment_encoded = sentiment_encoder.fit_transform(df[['sentiment']])
    sentiment_encoded = np.array(sentiment_encoded.toarray(), dtype=np.float32)
    additional_features = df[['text_length', 'word_count', 'stopword_ratio', 'punctuation_ratio', 'capitalization_ratio']].to_numpy()
    X = np.hstack((X_embeddings, sentiment_encoded, additional_features))
    model = model.fit(X)
    results[contam] = int(sum(model.predict(X)==-1))
    print(f"contam: {contam}, result: {sum(model.predict(X)==-1)}")

df = pd.read_csv('output/vectorized_data.csv')
for contam in [0.05, 0.15, 0.25, 0.35]:
    process(contam, df)

with open("output/metadata.json", "r+") as f: 
    metadata = json.load(f) 
    metadata["experiments"] = results 
    f.seek(0) 
    json.dump(metadata, f)