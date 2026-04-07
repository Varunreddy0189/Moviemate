import pandas as pd
import streamlit as st
import google.generativeai as genai

#API Key 
st.sidebar.subheader(" Enter Gemini API Key")

api_key = st.sidebar.text_input(
    "API Key",
    type="password",
    placeholder="Paste your Gemini API key here..."
)

if api_key:
    genai.configure(api_key=api_key)
    model_llm = genai.GenerativeModel("gemini-2.5-flash")
else:
    st.warning("Please enter your API key to start the chatbot.")
    st.stop()

model_llm = genai.GenerativeModel("gemini-2.5-flash")

    
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


import os
import numpy as np

if os.path.exists("movie_embeddings.npy"):
    embeddings = np.load("movie_embeddings.npy")
    print("✅ Loaded saved embeddings")
else:
    embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)
    np.save("movie_embeddings.npy", embeddings)
    print("✅ Created and saved embeddings")



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


    # Build movie context
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


    # Add conversation history (LAST 5 messages)
    history = ""
    for role, msg in st.session_state.chat_history[-5:]:
        prefix = "User" if role == "user" else "Assistant"
        history += f"{prefix}: {msg}\n"


    # Final prompt
    prompt = f"""
You are a conversational movie assistant like ChatGPT.

Conversation so far:
{history}

User's latest query:
{query}

Movies available:
{movies_context}

Instructions:
- Understand context from previous conversation
- Answer naturally like ChatGPT
- Recommend relevant movies
- Explain WHY
- Keep it conversational
"""

    response = model_llm.generate_content(prompt)
    return response.text


def update_memory(new_filters):
    for key, value in new_filters.items():
        if value is not None:
            st.session_state.chat_memory[key] = value

# add conversational logic





import re

def extract_filters(query):
    filters = {}
    q = query.lower()

    # actor detection
    actor_match = re.search(r"(actor|starring|with)\s([a-zA-Z\s]+)", q)
    if actor_match:
        filters["actor"] = actor_match.group(2).title().strip()

    # genre detection
    genres = ["action", "comedy", "drama", "thriller", "romance", "sci-fi", "adventure"]
    for g in genres:
        if g in q:
            filters["genre"] = g.capitalize()

    # year filters
    after_match = re.search(r'after (\d{4})', q)
    if after_match:
        filters["year_after"] = int(after_match.group(1))

    before_match = re.search(r'before (\d{4})', q)
    if before_match:
        filters["year_before"] = int(before_match.group(1))

    # rating
    if "high rated" in q or "top rated" in q:
        filters["min_rating"] = 8.0

    return filters


def apply_filters(df):
    memory = st.session_state.chat_memory
    filtered_df = df.copy()

    if memory["actor"]:
        filtered_df = filtered_df[
            filtered_df["Cast"].str.contains(memory["actor"], case=False)
        ]

    if memory["genre"]:
        filtered_df = filtered_df[
            filtered_df["Genre"].str.contains(memory["genre"], case=False)
        ]

    if memory["year_after"]:
        filtered_df = filtered_df[
            filtered_df["Year"] > memory["year_after"]
        ]

    if memory["year_before"]:
        filtered_df = filtered_df[
            filtered_df["Year"] < memory["year_before"]
        ]

    if memory["min_rating"]:
        filtered_df = filtered_df[
            filtered_df["Rating"] >= memory["min_rating"]
        ]

    return filtered_df


def chatbot(query, top_k=5):
    # extract filters
    new_filters = extract_filters(query)
    update_memory(new_filters)

    # apply filters
    filtered_df = apply_filters(df)

    #multi-turn context
    if len(st.session_state.chat_history) > 0:
        last_user_query = [
            msg for role, msg in reversed(st.session_state.chat_history)
            if role == "user"
        ][0]

        query = last_user_query + " " + query

    # filter-based response
    if len(new_filters) > 0:
        if len(filtered_df) == 0:
            return "No exact matches found."

        return generate_response(filtered_df, query)

    # semantic search fallback
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    if distances[0][0] > 1.5:
        return "Sorry, I couldn't find relevant movies."

    results = df.iloc[indices[0]]
    return generate_response(results, query)



# ---------------- session Memory ----------------
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = {
        "genre": None,
        "year_after": None,
        "year_before": None,
        "min_rating": None,
        "actor": None
    }


#UI streamlit

st.set_page_config(page_title="MovieMate", layout="wide")

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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.title("Controls")

top_k = st.sidebar.slider("Number of results", 1, 10, 5)

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []

st.title(" MovieMate")
st.markdown("Ask anything like: *'action movies after 2015'*")



def handle_submit():
    user_input = st.session_state.input_box

    if user_input:
        response = chatbot(user_input, top_k)

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response))

        st.session_state.input_box = ""


st.markdown("""
<style>
.bottom-input {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    width: 60%;
    z-index: 1000;
}
</style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="bottom-input">', unsafe_allow_html=True)

    
    st.text_input(
        "",
        key="input_box",
        placeholder="Ask about movies...",
        on_change=handle_submit,
        label_visibility="collapsed"
)

    st.markdown('</div>', unsafe_allow_html=True)




chat_container = st.container()

with chat_container:
    for role, content in st.session_state.chat_history:

        if role == "user":
            st.markdown(
                f'<div class="chat-user">🧑 {content}</div>',
                unsafe_allow_html=True
            )

        else:
            if isinstance(content, str):
                st.markdown(
                    f'<div class="chat-bot">🤖 {content}</div>',
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    '<div class="chat-bot">🤖 Here are some recommendations:</div>',
                    unsafe_allow_html=True
                )

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