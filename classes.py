import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from datetime import datetime
import pdb
import csv
import re
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import numpy as np
import openai


# base class for agent
class BaseAgent:
    """ Base class for all agents. """
    def __init__(self, model_name: str, api_key: str):
        self.llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key)

    def process(self, text: str):
        raise NotImplementedError("should be implemented in subclasses")
    
class ParaphraseAgent(BaseAgent):
    def process(self, text: str):
        #pdb.set_trace()
        response = self.llm.invoke([HumanMessage(content=f"Paraphrase this: '{text}'")])
        return response.content.strip()

class Augmenter_Agent:
    def __init__(self, paraphrase_agent: ParaphraseAgent, num_variations: int = 3):
        self.paraphrase_agent = paraphrase_agent
        self.num_variations = num_variations

    def augment_data(self, df: pd.DataFrame):
        synthetic_texts = []

        for id, text in enumerate(df['text']):
            for _ in range(self.num_variations):
                paraphrased_text = self.paraphrase_agent.process(text=text)
                synthetic_texts.append({
                    "id": len(df) + len(synthetic_texts) + 1,
                    "text": paraphrased_text,
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                })
        augmented_df = pd.concat([df, pd.DataFrame(synthetic_texts)], ignore_index=True)
        augmented_df.to_csv("output/augmented_data.csv", index=False, quoting=csv.QUOTE_ALL, sep=',')
        # Store metadata
        metadata = {
            "original_rows": len(df),
            "augmented_rows": len(augmented_df),
            "augmentation_ratio": len(augmented_df) / len(df),
            "num_variations_per_text": self.num_variations
        }
        with open("output/metadata.json", "w") as f:
            json.dump(metadata, f)
    

class NERAgent(BaseAgent):
    def process(self, text: str):
        #pdb.set_trace()
        response = self.llm.invoke([HumanMessage(content=f"Extract and give only entities without description in comma seperated format from this text: '{text}'. Give blank output if no entities are present.")])
        entities = [entity.strip() for entity in response.content.strip().split(',')]
        return entities
    
class sentimentAgent(BaseAgent):
    def find_sentiment(self, text):
        # Search for 'mixed' first (case-insensitive)
        if re.search(r"\bmixed\b", text, re.IGNORECASE):
            return "mixed"
        
        # Search for 'positive', 'negative', or 'neutral'
        match = re.search(r"\b(positive|negative|neutral)\b", text, re.IGNORECASE)
        
        # Return the matched sentiment in lowercase or None if not found
        return match.group(0).lower() if match else None

    def process(self, text: str):
        keywords = ["positive", "negative", "neutral"]
        response = self.llm.invoke([HumanMessage(content=f"Analyse sentiment and give output in either of positive, negative, neutral of the text: '{text}'")])
        sentiment = self.find_sentiment(response.content.strip())
        #print(text, response.content.strip(), sentiment)
        return sentiment

class themeAgent(BaseAgent):
    def process(self, text: str):
        response = self.llm.invoke([HumanMessage(content=f"Identify themes in comma seperated format in this text: '{text}'. Give out answer in one or two words.")])
        theme = [theme.strip() for theme in response.content.strip().split(',')]
        return theme
    
class anomalyAgent(BaseAgent):
    def process(self, df: pd.DataFrame):
        model = IsolationForest(contamination=0.2, random_state=99)
        X = np.vstack(df['embedding'].values)
        model.fit(X)
        df['anomaly_score'] = model.decision_function(X)
        df['is_anomaly'] = model.predict(X) == -1
        #pdb.set_trace()
        return df

class vectorizer(BaseAgent):
    def __init__(self, model_name, api_key):
        super().__init__(model_name, api_key)  # Initialize parent class if needed
        openai.api_key = api_key  # Set API key for OpenAI
        self.client = openai.OpenAI(api_key=openai.api_key)

    def get_tfidf_embedding(self, df):
        tfidf_vectorizer = TfidfVectorizer()
        embeddings = tfidf_vectorizer.fit_transform(df['text'])
        df['embedding'] = list(embeddings.toarray())
        df['embedding'] = df['embedding'].apply(np.array)

    def get_openAI_embedding(self, df):
        embeddings = [self.client.embeddings.create(model='text-embedding-3-small', input=text).data[0].embedding for text in df['text']]
        df['embedding'] = embeddings

    def process(self, df: pd.DataFrame):
        #self.get_tfidf_embedding(df)
        self.get_openAI_embedding(df)
        return df
    

class PipelineManager:
    def __init__(self, augmenter, ner_agent, sentiment_agent, theme_agent, vectorizer_agent, anomaly_agent):
        self.augmenter = augmenter
        self.ner_agent = ner_agent
        self.sentiment_agent = sentiment_agent
        self.theme_agent = theme_agent
        self.vectorizer_agent = vectorizer_agent
        self.anomaly_agent = anomaly_agent
    
    def run_pipeline(self, df: pd.DataFrame):
        print("Augmenting data...")
        self.augmenter.augment_data(df)

        df = pd.read_csv('output/augmented_data.csv')

        # step 2: Feature Extraction
        print("Extracting features...")
        df["NE"] = df['text'].apply(self.ner_agent.process)
        df["sentiment"] = df['text'].apply(self.sentiment_agent.process)
        df['theme'] = df['text'].apply(self.theme_agent.process)
        df.to_csv("output/features_data.csv", index=False, quoting=csv.QUOTE_ALL, sep=',')

        # step 3: vectorize the data and store vectorized_data.csv
        print("Vectorizing data...")
        df = pd.read_csv('output/features_data.csv')
        df = self.vectorizer_agent.process(df)
        df.to_csv("output/vectorized_data.csv", index=False, quoting=csv.QUOTE_ALL, sep=',')

        # step 3: Anomaly detection
        print("Detecting Anomaly...")
        # read vectorized data and run anomaly detection
        df = self.anomaly_agent.process(df)
        df.to_csv("output/anomaly_data.csv", index=False, quoting=csv.QUOTE_ALL, sep=',')
        return df
    
    def run_anomaly_detection(self):
        print("running anamoloy detection...")
        #pdb.set_trace()
        df = pd.read_csv('output/augmented_data.csv')
        df = self.anomaly_agent.process(df)
        return df

API_KEY = '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'
model_name = 'gpt-4o'

df = pd.read_csv("data/data.csv")

#agents
paraphrase_agent = ParaphraseAgent(model_name=model_name, api_key=API_KEY)
ner_agent = NERAgent(model_name=model_name, api_key=API_KEY)
sentiment_agent = sentimentAgent(model_name=model_name, api_key=API_KEY)
theme_agent = themeAgent(model_name=model_name, api_key=API_KEY)
vectorizer_agent = vectorizer(model_name=model_name, api_key=API_KEY)
anomaly_agent = anomalyAgent(model_name=model_name, api_key=API_KEY)


#augmenter and pipeline manager
augmenter_agent = Augmenter_Agent(paraphrase_agent=paraphrase_agent, num_variations=1)
pipeline = PipelineManager(augmenter=augmenter_agent, ner_agent=ner_agent, sentiment_agent=sentiment_agent, theme_agent=theme_agent, vectorizer_agent=vectorizer_agent, anomaly_agent=anomaly_agent)

augmented_df = pipeline.run_pipeline(df)
