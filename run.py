import pandas as pd
from classes import *

API_KEY = 'api_key'
model_name = 'gpt-4o'



#agents
paraphrase_agent = ParaphraseAgent(model_name=model_name, api_key=API_KEY)
ner_agent = NERAgent(model_name=model_name, api_key=API_KEY)
sentiment_agent = sentimentAgent(model_name=model_name, api_key=API_KEY)
theme_agent = themeAgent(model_name=model_name, api_key=API_KEY)
vectorizer_agent = vectorizer(model_name=model_name, api_key=API_KEY)
anomaly_agent = anomalyAgent(model_name=model_name, api_key=API_KEY)
textfeature_agent = textFeatureAgent(model_name=model_name, api_key=API_KEY)


#augmenter and pipeline manager
augmenter_agent = Augmenter_Agent(paraphrase_agent=paraphrase_agent, num_variations=10)
pipeline = PipelineManager(augmenter=augmenter_agent, ner_agent=ner_agent, sentiment_agent=sentiment_agent, theme_agent=theme_agent, vectorizer_agent=vectorizer_agent, anomaly_agent=anomaly_agent, textfeature_agent= textfeature_agent)

augmented_df = pipeline.run_pipeline('data/data.csv')