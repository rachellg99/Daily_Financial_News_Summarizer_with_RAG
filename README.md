# 📈 Daily Financial News Summarizer with RAG

This project implements a small-scale Retrieval-Augmented Generation (RAG) system to fetch, cluster, and summarize the day’s top financial news — with an interactive Q&A feature to answer follow-up questions about market highlights.

---

## Features

- **Automated News Summaries**  
  Fetches articles from CNBC, Yahoo Finance, Investing.com, and Google News; filters finance-relevant news; clusters them into five key topics; and generates concise headlines + summaries.

- **Retrieval-Augmented Q&A**  
  Embeds user questions and retrieves the most relevant topic summary to provide context-aware answers.

- **Fresh Daily Insights**  
  Caches article content with timestamps to ensure only today’s news is summarized.

- **Interactive Web UI**  
  Built with Streamlit for a simple, responsive interface.

---

## Technologies

- **LLM**: OpenAI’s `o4-mini` model  
- **Embeddings**: OpenAI `text-embedding-3-small`  
- **Clustering**: KMeans for topic grouping  
- **Frontend**: Streamlit web app  
- **Data sources**: Financial RSS feeds  

---

## Data Pipeline

1. **Fetch & Cache**: Collect today’s news from multiple RSS feeds; cache content with timestamps.
2. **Filter & Cluster**: Embed articles, filter by keywords, and cluster into 5 market topics.
3. **Summarize**: Generate concise headlines and 2-3 sentence summaries per topic with the LLM.
4. **Q&A (RAG)**: Embed user question, match to the best topic, and generate a contextual answer.

---

## Results

- A Streamlit app that displays **5 key market highlights** each day.
- Users can ask follow-up questions like:
  - *“What major banks were mentioned today?”*
  - *“Why did the stock market rally?”*
- The system responds with accurate, up-to-date answers based on retrieved articles.

---

## Future Improvements

- Integrate a vector database (e.g., Pinecone) for scalable, efficient retrieval.
- Add advanced clustering (e.g., HDBSCAN) for dynamic topic numbers.
- Replace static keywords with an NLP-based relevance classifier.
- Personalize topics based on user-selected industries or keywords.
- Add multi-language support and premium news sources for richer, more precise insights.

---

## Deployment

- **Local Deployment**  
  To run the app locally, execute:
  ```bash
  streamlit run app.py
