import pandas as pd
import pdb

def evaluate(df):
    true_anomalies = [12,13, 16, 5] # Hypothetical ground truth 
    predicted_anomalies = df[df["is_anomaly"]]["id"].tolist() 
    print(predicted_anomalies)
    precision = len(set(true_anomalies) & set(predicted_anomalies)) / len(predicted_anomalies) 
    recall = len(set(true_anomalies) & set(predicted_anomalies)) / len(true_anomalies) 
    print(f"Precision: {precision}, Recall: {recall}")

df = pd.read_csv('output/anomaly_data.csv')
evaluate(df)