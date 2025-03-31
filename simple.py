import openai

openai.api_key = "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"

client = openai.OpenAI(api_key=openai.api_key)

embeddings = client.embeddings.create(model='text-embedding-3-small', input='hello how are you doing')

print(embeddings.data[0].embedding)