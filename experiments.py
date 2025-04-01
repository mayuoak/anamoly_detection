from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import pdb
import ast
import json

results = {}
def process(contam, df: pd.DataFrame):
    model = IsolationForest(contamination=contam, random_state=99)
    X = np.vstack(df['embedding'].apply(ast.literal_eval).values)
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