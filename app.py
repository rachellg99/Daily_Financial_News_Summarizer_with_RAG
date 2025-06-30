import streamlit as st
import numpy as np
import json
import os
import requests
import feedparser
import re
from bs4 import BeautifulSoup
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min, cosine_similarity
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


CACHE_FILE = "article_cache.json"

# -------------------------
# NEWS FETCHING & FULL TEXT
# -------------------------
def load_cache():
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def fetch_full_article(url, pub_time_iso=None):
    cache = load_cache()
    if url in cache:
        cached_pub_time = cache[url].get("published")
        if cached_pub_time == pub_time_iso:
            return cache[url]["content"]
        # if the article's publish time is different from the cache file's time, means that the article has been updated → Fetch the news again

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        paras = [p.get_text() for p in soup.find_all("p")]
        paras = [p.strip() for p in paras if len(p.strip()) > 50]
        full_text = "\n".join(paras)
        # Save to cache
        cache[url] = {
            "content": full_text,
            "published": pub_time_iso
        }
        save_cache(cache)
        return full_text
    except Exception:
        return ""


@st.cache_data
def fetch_news():
    rss_feeds = [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC US Top News
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EDJI&region=US&lang=en-US",  # Yahoo Finance
        "https://www.investing.com/rss/news_25.rss",  # Investing.com
        "https://news.google.com/rss/search?q=finance",  # Google News finance
    ]
    all_news = []
    for feed in rss_feeds:
        parsed = feedparser.parse(feed)
        for entry in parsed.entries[:5]:
            # Get the publish time
            pub_time = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_time = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                pub_time = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

            if pub_time is None:
                continue  # If no timestamp → skip that article

            # Only save the article that is published today
            if pub_time.date() != datetime.utcnow().date():
                continue

            pub_time_iso = pub_time.isoformat()
            content = fetch_full_article(entry.link, pub_time_iso=pub_time_iso)
            news_item = {
                "title": entry.title,
                "source": parsed.feed.title if 'title' in parsed.feed else feed,
                "content": content if content else entry.get("summary", ""),
                "link": entry.link
            }
            all_news.append(news_item)

    with open("news.json", "w", encoding="utf-8") as f:
        json.dump(all_news, f, ensure_ascii=False, indent=2)
    return all_news

# -------------------------
# EMBEDDING + SUMMARIZATION
# -------------------------
def embed_text(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return np.array(resp.data[0].embedding)

@st.cache_data
def summarize_5_key_topics(news_list, keywords=None, n_clusters=5):
    if keywords is None:
        keywords = ["finance", "stock", "market", "inflation", "interest", "oil", "economy", "fed"]

    filtered = []
    for n in news_list:
        text = (n["title"] + " " + n["content"]).lower()
        if any(kw in text for kw in keywords):
            filtered.append(n)
    if not filtered:
        return []

    embeddings = [embed_text(news["title"] + " " + news["content"][:1000]) for news in filtered]
    embeddings = np.vstack(embeddings)
    k = min(n_clusters, len(filtered))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    topics = []
    for i in range(k):
        idxs = np.where(labels == i)[0]
        cluster_news = [filtered[idx] for idx in idxs]

        # Check whether the news in the same cluster share the same theme
        check_prompt = "You are a financial editor. Here are news headlines and their summaries:\n\n"
        for idx, news in enumerate(cluster_news):
            snippet = news['content'][:200].replace("\n", " ")
            check_prompt += f"{idx+1}) Title: {news['title']}\nSummary: {snippet}\n\n"
        check_prompt += (
            "Do these articles belong to the same market topic? Answer 'Yes' or 'No'. "
            "If 'No', list the article numbers that do not belong, e.g., 'No. Articles: 2, 4'."
        )
        check_resp = client.chat.completions.create(
            model="o4-mini",
            messages=[{"role": "user", "content": check_prompt}]
        )
        check_ans = check_resp.choices[0].message.content.strip().lower()

        if "no" in check_ans:
            unmatched = re.findall(r"\d+", check_ans)
            unmatched_idx = [int(i)-1 for i in unmatched if int(i)-1 in range(len(cluster_news))]
            cluster_news = [n for i, n in enumerate(cluster_news) if i not in unmatched_idx]

        if len(cluster_news) == 0:
            continue  # Skip the clusters that do not contain any news


        # Generate the summary for each cluster
        combined = ""
        for n, news in enumerate(cluster_news[:3]):
            combined += f"Article {n+1} Title: {news['title']}\nContent: {news['content'][:1500]}\n\n"
        prompt = (
            "You are a professional financial journalist.\n\n"
            "Based on the following articles, you need to do these tasks strictly in order:\n\n"
            "1) Generate exactly one short, catchy topic headline (6-12 words) that summarizes the core theme across all articles.\n\n"
            "2) Then write a concise 2-3 sentence summary in English describing the topic details without repeating the headline or its exact words.\n\n"
            "Important output rules:\n"
            "- Return your answer in exactly this format:\n"
            "Headline: [your short headline here]\n"
            "Summary: [your 2-3 sentence summary here, do not restate the headline]\n"
            "- Do not include any text before or after this format.\n"
            "- Do not repeat the headline inside the summary.\n\n"
            "Here are the articles:\n"
            "---\n"
            f"{combined}\n"
            "---"
        )
        resp = client.chat.completions.create(
            model="o4-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        output = resp.choices[0].message.content.strip()

        # Use regex to extract headline
        headline_match = re.search(r"(Headline:)?\s*(.+?)\s*(?=Summary:|$)", output, re.IGNORECASE | re.DOTALL)
        summary_match = re.search(r"(Summary:)\s*(.+)", output, re.IGNORECASE | re.DOTALL)

        if headline_match:
            headline = headline_match.group(2).replace("*", "").strip()
        else:
            headline = "Untitled Topic"

        if summary_match:
            summary = summary_match.group(2).strip()
        else:
            summary = output.strip()

        try:
            if len(cluster_news) > 0:
                center_emb = kmeans.cluster_centers_[i].reshape(1, -1)
                closest_idxs, _ = pairwise_distances_argmin_min(center_emb, embeddings[idxs])
                if closest_idxs.size > 0 and closest_idxs[0] < len(cluster_news):
                    rep_links = [cluster_news[closest_idxs[0]]['link']]
                else:
                    rep_links = [cluster_news[0]['link']]
            else:
                # Skip cluster entirely if it ended up empty
                continue
        except Exception as e:
            print(f"[Warning] Could not find representative link for cluster {i}: {e}")
            rep_links = []

        topics.append({
            "headline": headline,
            "summary": summary,
            "links": rep_links,
            "content": " ".join([n['content'] for n in cluster_news[:3]]),
        })

    # If the final number of topics is fewer than 5, add additional topics using remaining filtered news:
    # - Calculate how many more topics are needed (needed)
    # - Identify filtered news articles not already included in topics
    # - Append simple topics built from these remaining articles (using title and a short content snippet) until there are at least 3 topics
    if len(topics) < 5:
        needed = 5 - len(topics)
        used_contents = set(t['content'] for t in topics)
        remaining_news = [n for n in filtered if n['content'] not in used_contents]
        for news in remaining_news[:needed]:
            topics.append({
                "headline": news["title"],
                "summary": news["content"][:300],
                "links": [news["link"]],
                "content": news["content"],
            })

    return topics

# -------------------------
# UI LOGIC
# -------------------------

today = datetime.utcnow().strftime("%Y-%m-%d")

st.markdown(
    f"<h3 style='margin-bottom: 1rem;'>📈 News Breakfast ({today}): Top 5 Market Highlights with Q&A</h3>",
    unsafe_allow_html=True,
)

if st.button("🔄 Fetch & Summarize Today's News"):
    with st.spinner("Fetching and summarizing news..."):
        news_list = fetch_news()
        key_topics = summarize_5_key_topics(news_list)
        st.session_state["key_topics"] = key_topics
        st.success("News updated!")

if "key_topics" in st.session_state:
    for i, t in enumerate(st.session_state["key_topics"], 1):
        st.markdown(
            f"<div style='font-size:22px; font-weight:700;'>{i}) {t['headline']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:18px;'><b>Summary:</b> {t['summary']}</div>",
            unsafe_allow_html=True,
        )
        with st.expander("Related News Links"):
            for link in t["links"]:
                st.markdown(f"- [{link}]({link})")

    st.markdown("---")

    # Build embeddings for RAG
    db = [{"embedding": embed_text(t["content"]), "topic": t} for t in st.session_state["key_topics"]]

    # st.subheader("❓Ask follow-up question about today's news")
    st.markdown("""
            <div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px;'>
            <h4>❓ Ask follow-up question about today's news</h4>
            </div>
    """, unsafe_allow_html=True)
    
    question = st.text_input("Type your question here:")
    if st.button("Get Answer"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            q_emb = embed_text(question)
            sims = [cosine_similarity(q_emb.reshape(1,-1), item["embedding"].reshape(1,-1))[0,0] for item in db]
            best_idx = int(np.argmax(sims))
            best_topic = db[best_idx]["topic"]
            prompt = (
                f"Question: {question}\n\n"
                f"Based on this topic, provide a concise answer in English:\n"
                f"Headline: {best_topic['headline']}\nSummary: {best_topic['summary']}\nContent: {best_topic['content']}\n"
            )
            response = client.chat.completions.create(
                model="o4-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content.strip()
            st.success("Answer:")
            st.write(answer)
else:
    st.info("Press the button above to fetch and summarize today's news.")
