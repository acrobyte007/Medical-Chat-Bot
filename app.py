import streamlit as st
import pickle
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer('all-MiniLM-L6-v2')

def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def send_to_gemini_api(output_text, api_key):
    output_text="give the summary of it like human :"+output_text
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    headers = {
        'Content-Type': 'application/json',
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": output_text}
                ]
            }
        ]
    }

    response = requests.post(api_url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}, {response.text}"

def find_corresponding_output(user_query, data, api_key):
    input_vectors = np.array([entry['embedding'] for entry in data])
    output_texts = [entry['output'] for entry in data]

    query_embedding = model.encode([user_query])[0]

    similarities = cosine_similarity([query_embedding], input_vectors)[0]
    most_similar_idx = np.argmax(similarities)
    corresponding_output = output_texts[most_similar_idx]

    structured_response = send_to_gemini_api(corresponding_output, api_key)
    return structured_response

def display_structured_response(response):
    if isinstance(response, dict):
        content = response.get('candidates', [])[0].get('content', {}).get('parts', [])[0].get('text', '')
        st.write("**Model's Response:**")
        st.markdown(content)


    else:
        st.error("Error fetching response from the API.")

# Streamlit UI
def main():
    st.title("Detection Query Interface")


    embeddings_file_path = 'embeddings_output.pkl'
    data = load_pickle(embeddings_file_path)
    st.success("Embeddings loaded successfully!")

    user_query = st.text_input("Enter your query:")

    api_key = 'your_API_key'

    if st.button("Submit"):
        if user_query:

            structured_answer = find_corresponding_output(user_query, data, api_key)
            display_structured_response(structured_answer)
        else:
            st.error("Please enter a query.")

if __name__ == '__main__':
    main()
