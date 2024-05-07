import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV data into a DataFrame
df = pd.read_csv("CleanLatest.csv")

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the document titles
tfidf_matrix = vectorizer.fit_transform(df[['comments']])

def brainfish_medical_assistant_prompt(user_query, retrieved_text):
    prompt = f"""
    You are Brainfish, an AI medical assistant expert. Your primary function is to provide accurate medical information when queried. Follow these structured steps to ensure that your responses are appropriate and relevant:

    1. Assess Query Relevance: Determine if the user's question pertains to medical topics. This step ensures that you stay within your designated expertise
    
    2. Predict the medical speciality of {user_query}

    2. Context Verification: Check if the necessary information to answer the query is available within the provided context. This may include symptoms, medical conditions, treatment options. Don't include anyother people condition. only rely on fatts

    Here is the context:
    {retrieved_text}

    Based on context and medical speciality, please answer the following user query:
    {user_query}

    Remember to respond in a professional, doctor-like tone.
    """
    return prompt

def retrieve_documents(query, top_n=3):
    # Transform the query into a vector
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity between the query vector and document vectors
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

    # Get the indices of the top-n most similar documents
    top_indices = similarity_scores.argsort()[0][-top_n:][::-1]

    # Retrieve the titles of the top-n most similar documents
    top_titles = df.iloc[top_indices]["title"].tolist()

    return " ".join(top_titles)

def generate_response(query):
    # Retrieve relevant documents from the CSV
    retrieved_text = retrieve_documents(query)

    # Prepare the input for the model using the Brainfish medical assistant prompt
    prompt = brainfish_medical_assistant_prompt(query, retrieved_text)

    # Make a request to the Mistral API
    endpoint = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral",
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False
    }
    response = requests.post(endpoint, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        generated_text = result["response"]
        return generated_text
    else:
        return f"Error: {response.status_code}"

# Streamlit app setup

def main():
    st.title("Brainfish Medical Assistant")

    query = st.text_input("How can i help you? Write your symptoms")

    if st.button("Consult"):
        if query.strip() != "":
            with st.spinner("Generating response..."):
                response = generate_response(query)
            st.write("Response:")
            st.markdown(f"<div style='animation: fadeIn 2s;'>{response}</div>", unsafe_allow_html=True)
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()