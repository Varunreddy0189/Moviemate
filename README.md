
#  MovieMate AI  
### Intelligent Movie Recommendation Chatbot using NLP + LLM

MovieMate AI is an NLP-powered movie recommendation system that combines semantic search, filter-based retrieval, and LLM-generated responses to deliver intelligent, conversational movie suggestions.

---

## Features

- ## Semantic Search
  - Uses Sentence Transformers + FAISS
  - Finds movies based on meaning (not just keywords)

- ## Optimized Embeddings
  - Auto-generates on first run  
  - Loads instantly in future runs  

- ## Advanced Filtering
  - Genre (Action, Comedy, etc.)
  - Actor-based search (e.g., "movies with Tom Cruise")
  - Year filters (before/after)
  - Rating filters (top-rated)

- ## Conversational Chatbot
  - Handles follow-up queries (e.g., "only after 2015")
  - Maintains chat memory  

- ## LLM Integration (Google Gemini)
  - Generates human-like responses  
  - Explains why movies are recommended  

- ## Streamlit Web App
  - Interactive chat UI  
  - Movie cards with details  

- ## Data Visualization
  - Rating distribution using Matplotlib    

---

## Project Structure

├── app.py
├── data
│   └── imdb_top_1000.csv
├── movie_embeddings.npy
├── README.md
└── requirements.txt

## installation

## 1. Clone the Repository
    ```bash
    git clone https://github.com/Varunreddy0189/Moviemate.git
    cd Moviemate
    ```

## 2. install dependencies

    pip install -r requirements.txt

## 3. add API key

    open app.py and replace the API key 
    genai.configure(api_key="YOUR_API_KEY")

## 4. run the application

    streamlit run app.py


## How It Works

### 1. Data Preprocessing
- Clean the dataset  
- Convert runtime and year into proper format  
- Combine features (title, genre, summary, director, cast) into a single text column  

### 2. Embedding Generation
- Uses `all-MiniLM-L6-v2` (Sentence Transformers)  
- Convert each movie into a numerical vector (embedding)  

### 3. FAISS Similarity Search
- Store embeddings in a FAISS index  
- Finds movies similar to user query using vector distance  

### 4. Filter Extraction
- Extracts structured filters from user input using regex  
- Supports:
  - Genre  
  - Actor
  - Year   
  - Rating   

### 5. Filtering + Retrieval
- Applies filters to dataset (strict matching)  
- If no filters → performs semantic search  

### 6. Response Generation (LLM)
- Sends retrieved movies to Gemini API ands generates conversational recommendations
- Explains why movies match user query  

### 7. Streamlit Interface
- Displays chatbot interaction. Shows movie cards with details (title, rating, cast, etc.)  