

import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

df = pd.read_csv('/content/chat_data.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')

if 'input' not in df.columns or 'output' not in df.columns:
    raise ValueError("CSV file must contain 'input' and 'output' columns")


input_texts = df['input'].tolist()
input_embeddings = model.encode(input_texts)


embedding_data = [{"embedding": embedding, "output": output} for embedding, output in zip(input_embeddings, df['output'].tolist())]

with open('embeddings_output.pkl', 'wb') as f:
    pickle.dump(embedding_data, f)
