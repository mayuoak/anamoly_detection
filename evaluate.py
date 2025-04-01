import pandas as pd
import pdb

def evaluate(df):
    true_anomalies = [3,4,5,8,9] # Hypothetical ground truth
    GT = true_anomalies.copy()
    # extrapolate anomalies by adding 10 to each id
    print(len(df))
    for i in range(1, len(df)//10):
        extrapolate_anomalies = [x+(10*i) for x in true_anomalies]
        GT.extend(extrapolate_anomalies) 
    predicted_anomalies = df[df["is_anomaly"]]["id"].tolist() 
    print(predicted_anomalies)
    precision = len(set(GT) & set(predicted_anomalies)) / len(predicted_anomalies) 
    recall = len(set(GT) & set(predicted_anomalies)) / len(GT) 
    print(f"Precision: {precision}, Recall: {recall}")

df = pd.read_csv('output/anomaly_data.csv')
evaluate(df)