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
import ast
import os

api_calls = 0

# base class for agent
class BaseAgent:
    """ Base class for all agents. """
    def __init__(self, model_name: str, api_key: str):
        self.llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key)
        self.api_calls = 0
        self.batch_size = 3

    def process(self, text: str):
        raise NotImplementedError("should be implemented in subclasses")
    
    def get_response(self, texts: str):
        #pdb.set_trace()
        global api_calls
        batch_messages = [HumanMessage(content=text) for text in texts]
        responses = self.llm.batch(texts)
        api_calls += 1
        self.update_metadata()
        return responses
    
    def update_metadata(self):
        #pdb.set_trace()
        if not os.path.exists("output/metadata.json"):
            return
        with open("output/metadata.json", "r+") as f:
            metadata = json.load(f)
            metadata["api_calls"] = api_calls
            f.seek(0)
            json.dump(metadata, f)
            f.truncate()
        
    
class ParaphraseAgent(BaseAgent):
    def process(self, texts: str):
        #pdb.set_trace()
        responses = self.get_response(texts)
        _responses = [re.sub(r'^["\']+|["\']+$', '', response.content.strip()) for response in responses] # sometimes model gives extra quotes thinking this is quoted sentence
        return _responses

class Augmenter_Agent:
    def __init__(self, paraphrase_agent: ParaphraseAgent, num_variations: int = 3, batch_size: int = 3):
        self.paraphrase_agent = paraphrase_agent
        self.num_variations = num_variations
        self.batch_size = batch_size

    def augment_data(self, df: pd.DataFrame):
        synthetic_texts = []
        texts = df['text'].tolist()
        # Process in batches
        for _ in range(self.num_variations):
            for i in range(0, len(texts), self.batch_size):
                batch_texts = [f"Paraphrase this: '{text}'" for text in texts[i:i + self.batch_size]]
                responses = self.paraphrase_agent.get_response(batch_texts)
                for response in responses:
                    paraphrased_text = response.content.strip()
                    synthetic_texts.append({
                        "id": len(df) + len(synthetic_texts) + 1,
                        "text": paraphrased_text.strip(),
                        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S").strip()
                    })
            # Process remaining texts
            remaining_texts = [f"Paraphrase this: '{text}'" for text in texts[i:i + self.batch_size]]
            responses = self.paraphrase_agent.get_response(remaining_texts)
            for response in responses:
                paraphrased_text = response.content.strip()
                synthetic_texts.append({
                    "id": len(df) + len(synthetic_texts) + 1,
                    "text": paraphrased_text.strip(),
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S").strip()
                })
        
        augmented_df = pd.concat([df, pd.DataFrame(synthetic_texts)], ignore_index=True)
        
        # Store metadata
        metadata = {
            "original_rows": len(df),
            "augmented_rows": len(augmented_df),
            "augmentation_ratio": len(augmented_df) / len(df),
            "num_variations_per_text": self.num_variations, 
            "api_calls": self.paraphrase_agent.api_calls
        }
        
        return augmented_df, metadata
        
class NERAgent(BaseAgent):
    def process(self, texts: str):
        #pdb.set_trace()
        entities = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = [f"Extract and give only entities without description in comma seperated format from this text: '{text}'. Give blank output if no entities are present." for text in texts[i:i + self.batch_size]]
            responses = self.get_response(batch_texts)
            entities.extend([response.content.strip() for response in responses])
        # Process remaining texts
        remaining_texts = texts[len(texts) - len(texts) % self.batch_size:]
        batch_texts = [f"Extract and give only entities without description in comma seperated format from this text: '{text}'. Give blank output if no entities are present." for text in remaining_texts]
        responses = self.get_response(batch_texts)
        #_responses = [re.sub(r'^["\']+|["\']+$', '', response.content.strip()) for response in responses] # sometimes model gives extra quotes thinking this is quoted sentence
        entities.extend([response.content.strip() for response in responses])
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

    def process(self, texts: str):
        keywords = ["positive", "negative", "neutral"]
        sentiments = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = [f"Analyse sentiment and give output in either of positive, negative, neutral of the text: '{text}'" for text in texts[i:i + self.batch_size]]
            responses = self.get_response(batch_texts)
            _responses = [re.sub(r'^["\']+|["\']+$', '', response.content.strip()) for response in responses] # sometimes model gives extra quotes thinking this is quoted sentence
            _sentiments = [self.find_sentiment(response) for response in _responses]
            sentiments.extend(_sentiments)
        # Process remaining texts   
        remaining_texts = texts[len(texts) - len(texts) % self.batch_size:]
        batch_texts = [f"Analyse sentiment and give output in either of positive, negative, neutral of the text: '{text}'" for text in remaining_texts]
        responses = self.get_response(batch_texts)
        _sentiments = [self.find_sentiment(response.content.strip()) for response in responses]
        sentiments.extend(_sentiments)
        return sentiments

class themeAgent(BaseAgent):
    def process(self, text: str):
        themes = []
        for i in range(0, len(text), self.batch_size):
            batch_texts = [f"Identify themes in comma seperated format in this text: '{text}'." for text in text[i:i + self.batch_size]]
            responses = self.get_response(batch_texts)
            _responses = [re.sub(r'^["\']+|["\']+$', '', response.content.strip()) for response in responses] # sometimes model gives extra quotes thinking this is quoted sentence
            themes.extend([res.split(',') for res in _responses])
        # Process remaining texts
        remaining_texts = text[len(text) - len(text) % self.batch_size:]
        batch_texts = [f"Identify themes in comma seperated format in this text: '{text}'." for text in remaining_texts]
        responses = self.get_response(batch_texts)
        _responses = [re.sub(r'^["\']+|["\']+$', '', response.content.strip()) for response in responses] # sometimes model gives extra quotes thinking this is quoted sentence
        themes.extend([res.split(',') for res in _responses])
        return themes
    
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
        global api_calls
        texts = df['text'].tolist()
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.client.embeddings.create(
                model='text-embedding-3-small', 
                input=batch_texts)
            api_calls += 1
            
            embeddings.extend([embedding.embedding for embedding in batch_embeddings.data])

        # Process remaining texts
        remaining_texts = texts[len(texts) - len(texts) % self.batch_size:]
        if len(remaining_texts) > 0:
            #pdb.set_trace()
            batch_embeddings = self.client.embeddings.create(
                model='text-embedding-3-small',
                input=remaining_texts)
            api_calls += 1
            embeddings.extend([embedding.embedding for embedding in batch_embeddings.data])
        self.update_metadata()
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
    
    def run_pipeline(self, data):
        print("Augmenting data...")
        df = pd.read_csv(data)
        augmented_df, metadata = self.augmenter.augment_data(df)
        augmented_df.to_csv("output/augmented_data.csv", index=False, quoting=csv.QUOTE_ALL, sep=',')
        with open("output/metadata.json", "w") as f:
            json.dump(metadata, f)


        # step 2: Feature Extraction
        print("Extracting features...")
        df = pd.read_csv('output/augmented_data.csv')
        df["NE"] = self.ner_agent.process(df['text'].tolist())
        df["sentiment"] = self.sentiment_agent.process(df['text'].tolist())
        df['theme'] = self.theme_agent.process(df['text'].tolist())
        df.to_csv("output/features_data.csv", index=False, quoting=csv.QUOTE_ALL, sep=',')

        # step 3: vectorize the data and store vectorized_data.csv
        print("Vectorizing data...")
        df = pd.read_csv('output/features_data.csv')
        df = self.vectorizer_agent.process(df)
        df.to_csv("output/vectorized_data.csv", index=False, quoting=csv.QUOTE_ALL, sep=',')

        # step 3: Anomaly detection
        print("Detecting Anomaly...")
        # read vectorized data and run anomaly detection
        df = pd.read_csv('output/vectorized_data.csv')
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
        df = self.anomaly_agent.process(df)
        df.to_csv("output/anomaly_data.csv", index=False, quoting=csv.QUOTE_ALL, sep=',')
        return df
    
    def run_anomaly_detection(self):
        print("running anamoloy detection...")
        #pdb.set_trace()
        df = pd.read_csv('output/augmented_data.csv')
        df = self.anomaly_agent.process(df)
        return df


