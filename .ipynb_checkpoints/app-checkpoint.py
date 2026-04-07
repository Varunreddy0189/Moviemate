import pandas as pd

import google.generativeai as genai

genai.configure(api_key="AIzaSyA_i-zo9fJrmDMorj8OI1MPICbvccNJ6RU")

model_llm = genai.GenerativeModel("gemini-1.0-pro")

import google.generativeai as genai

for m in genai.list_models():
    print(m.name)
    
df = pd.read_csv("data/imdb_top_1000.csv")
df.head()


print(df.columns)


df["Runtime"] = df["Runtime"].astype(str)
df["Runtime"] = df["Runtime"].str.extract('(\d+)')
df["Runtime"] = df["Runtime"].astype(int)


df["Cast"] = df["Star1"] + ", " + df["Star2"] + ", " + df["Star3"] + ", " + df["Star4"]

df = df.rename(columns={
    "Poster_Link":"Poster",
    "Series_Title": "Title",
    "Released_Year": "Year",
    "IMDB_Rating": "Rating",
    "Runtime": "Duration",
    "Overview": "Summary"
})

df = df[[
    "Title", "Year", "Genre", "Rating",
    "Director", "Cast", "Duration", "Summary"
]]

df = df.dropna()


df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)


df["combined_text"] = (
    df["Title"] + " " +
    df["Genre"] + " " +
    df["Summary"] + " " +
    df["Director"] + " " +
    df["Cast"]
)

import matplotlib.pyplot as plt

plt.figure()
plt.hist(df["Rating"], bins=20)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Movies")
plt.show()


from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

df["combined_text"].head()     # checking if the combined text is working or not


embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)
print(embeddings.shape)


import numpy as np
np.save("movie_embeddings.npy", embeddings)



import faiss
import numpy as np

embeddings = np.load("movie_embeddings.npy")
embeddings = embeddings.astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


print(index.ntotal)


def search_movies(query, top_k=5):
    query_vector = model.encode([query]).astype("float32")
    
    distances, indices = index.search(query_vector, top_k)
    
    results = df.iloc[indices[0]]
    
    return results[["Title", "Genre", "Rating", "Year"]]


chat_memory = {
    "last_query": "",
    "filters": {}
}


# improve search function 

def retrieve_movies(query, top_k=5):
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    results = df.iloc[indices[0]]
    return results

#build response generator LLM like





def generate_response(results, query):

    if isinstance(results, str):
        return results

    if len(results) == 0:
        return "I couldn't find any movies matching your request."

    # 🔹 Convert results into context
    movies_context = ""
    for _, row in results.iterrows():
        movies_context += f"""
Title: {row['Title']}
Year: {row['Year']}
Genre: {row['Genre']}
Rating: {row['Rating']}
Director: {row['Director']}
Cast: {row['Cast']}
Summary: {row['Summary']}
---
"""

    # 🔹 Prompt
    prompt = f"""
You are an intelligent movie recommendation assistant.

User query:
{query}

Movies available in database:
{movies_context}

Instructions:
- Understand the user's intent (story, actor, genre, mood, etc.)
- Recommend ONLY relevant movies from the list
- Explain WHY they match the query
- If nothing matches well, say: "No relevant movies found in database"
- Respond conversationally like ChatGPT
"""

    response = model_llm.generate_content(prompt)

    return response.text


def update_memory(new_filters):
    global chat_memory
    for key, value in new_filters.items():
        if value is not None:
            chat_memory[key] = value

# add conversational logic

def chatbot(query):
    global chat_memory

    # If user refines query
    if "after" in query or "before" in query:
        full_query = chat_memory["last_query"] + " " + query
    else:
        full_query = query
        chat_memory["last_query"] = query

    results = retrieve_movies(full_query)

    response = generate_response(results, full_query)

    return response


print(chatbot("Suggest action movies"))
print(chatbot("only after 2015"))




chat_memory = {
    "genre": None,
    "year_after": None,
    "year_before": None,
    "min_rating": None,
    "actor": None
}

import re

def extract_filters(query):
    filters = {}
    q = query.lower()

    # 🎭 Actor detection
    actor_match = re.search(r"(actor|starring|with)\s([a-zA-Z\s]+)", q)
    if actor_match:
        filters["actor"] = actor_match.group(2).title().strip()

    # 🎬 Genre detection
    genres = ["action", "comedy", "drama", "thriller", "romance", "sci-fi", "adventure"]
    for g in genres:
        if g in q:
            filters["genre"] = g.capitalize()

    # 📅 Year filters
    after_match = re.search(r'after (\d{4})', q)
    if after_match:
        filters["year_after"] = int(after_match.group(1))

    before_match = re.search(r'before (\d{4})', q)
    if before_match:
        filters["year_before"] = int(before_match.group(1))

    # ⭐ Rating
    if "high rated" in q or "top rated" in q:
        filters["min_rating"] = 8.0

    return filters


def apply_filters(df):
    filtered_df = df.copy()

    if chat_memory["actor"]:
        filtered_df = filtered_df[
            filtered_df["Cast"].str.contains(chat_memory["actor"], case=False)
        ]

    if chat_memory["genre"]:
        filtered_df = filtered_df[
            filtered_df["Genre"].str.contains(chat_memory["genre"], case=False)
        ]

    if chat_memory["year_after"]:
        filtered_df = filtered_df[
            filtered_df["Year"] > chat_memory["year_after"]
        ]

    if chat_memory["year_before"]:
        filtered_df = filtered_df[
            filtered_df["Year"] < chat_memory["year_before"]
        ]

    if chat_memory["min_rating"]:
        filtered_df = filtered_df[
            filtered_df["Rating"] >= chat_memory["min_rating"]
        ]

    return filtered_df


def chatbot(query, top_k=5):
    new_filters = extract_filters(query)
    update_memory(new_filters)

    filtered_df = apply_filters(df)

    # 🔥 If filters exist → strict filtering
    if len(new_filters) > 0:
        if len(filtered_df) == 0:
            return "No exact matches found."

        return generate_response(results, query)

    # 🔥 Else → semantic search (story / vague query)
    query_vector = model.encode([query]).astype("float32")

    distances, indices = index.search(query_vector, top_k)

# 🔥 Check relevance (lower distance = better match)
    if distances[0][0] > 1.5:   # threshold (tune if needed)
        return "Sorry, I couldn't find any movies matching your request."

    results = df.iloc[indices[0]]
    return generate_response(results, query)



import streamlit as st

# ---------------- Page Config ----------------
st.set_page_config(page_title="MovieMate", layout="wide")

# ---------------- Styling ----------------
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }

.chat-user {
    background-color: #1f77b4;
    padding:10px;
    border-radius:10px;
    margin:5px;
    color:white;
}

.chat-bot {
    background-color: #262730;
    padding:10px;
    border-radius:10px;
    margin:5px;
    color:white;
}

.movie-card {
    background-color: #1c1f26;
    padding:15px;
    border-radius:15px;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- Sidebar ----------------
st.sidebar.title("🎛 Controls")

top_k = st.sidebar.slider("Number of results", 1, 10, 5)

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

# ---------------- Header ----------------
st.title("🎬 MovieMate AI")
st.markdown("Ask anything like: *'action movies after 2015'*")

# ---------------- Input ----------------
user_input = st.text_input("Enter your query:")

# ---------------- Chat Logic ----------------
if user_input:
    results = chatbot(user_input, top_k)

    st.session_state.chat_history.append(("user", user_input))

    if len(results) == 0:
        st.session_state.chat_history.append(("bot", "No movies found."))
    else:
        st.session_state.chat_history.append(("bot", results))

# ---------------- Chat Display ----------------
for role, content in st.session_state.chat_history:

    if role == "user":
        st.markdown(f'<div class="chat-user">🧑 {content}</div>', unsafe_allow_html=True)

    else:
        if isinstance(content, str):
            st.markdown(f'<div class="chat-bot">🤖 {content}</div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="chat-bot">🤖 Here are some recommendations:</div>', unsafe_allow_html=True)

            for _, row in content.iterrows():
                col1, col2 = st.columns([1, 4])

                with col1:
                    if "Poster" in row and pd.notna(row["Poster"]):
                        st.image(row["Poster"], use_container_width=True)

                with col2:
                    st.markdown(f"""
<div class="movie-card">
<b>🎬 {row['Title']} ({row['Year']})</b><br>
⭐ Rating: {row['Rating']}<br>
🎭 Genre: {row['Genre']}<br>
🎬 Director: {row['Director']}<br>
👨‍🎤 Cast: {row['Cast']}<br>
⏱ Duration: {row['Duration']} min
</div>
""", unsafe_allow_html=True)