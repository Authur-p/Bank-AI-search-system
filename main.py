import os
import json
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openai import AzureOpenAI

load_dotenv()


#  FILE LOADER (PDF + TEXT FILES) 

def load_file(path):
    if path.lower().endswith(".pdf"):
        text = ""
        with open(path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return text

    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


#  AZURE CLIENT SETUP 

wema_analyst = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("API_KEY")
)


#  LOAD ALL DOCUMENTS FROM /data FOLDER 

DATA_FOLDER = "data"
docs = {}

for file in os.listdir(DATA_FOLDER):
    full_path = os.path.join(DATA_FOLDER, file)
    if os.path.isfile(full_path):
        doc_name = file.split(".")[0]
        docs[doc_name] = lambda p=full_path: load_file(p)

print(f"Loaded {len(docs)} documents from /data")



# EMBEDDING CACHE FUNCTIONS 

CACHE_FILE = "embeddings_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

def file_last_modified(path):
    return int(os.path.getmtime(path))



#  GENERATE OR LOAD EMBEDDINGS

cache = load_cache()
embeddings = {}

print("Generating or loading embeddings...")

for name, loader in docs.items():

    file_path = loader.__defaults__[0]   # path from lambda
    modified_time = file_last_modified(file_path)

    # If cached AND file unchanged, load from cache
    if name in cache and cache[name]["last_modified"] == modified_time:
        print(f"Loaded from cache: {name}")
        embeddings[name] = np.array(cache[name]["embedding"])
        continue

    # Else re-embed
    print(f"Embedding fresh: {name}")
    text = loader()

    res = wema_analyst.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    embed = np.array(res.data[0].embedding)
    embeddings[name] = embed

    # Save to cache
    cache[name] = {
        "embedding": embed.tolist(),
        "source_file": file_path,
        "last_modified": modified_time
    }

# Save updated cache
save_cache(cache)
print("Embedding caching complete.\n")



#  COSINE SIMILARITY FUNCTION 

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


#  CONTINUOUS QUESTION LOOP 

while True:
    question = input("\nAsk your question (or type 'quit'): ")

    if question.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    # Embed the question
    q_embed = np.array(
        wema_analyst.embeddings.create(
            input=question,
            model="text-embedding-3-small"
        ).data[0].embedding
    )
    # Compute similarity with each document
    scores = []
    for name, emb in embeddings.items():
        score = cosine_similarity(q_embed, emb)
        scores.append((name, score))

    # Get top 3 most relevant documents
    top3 = sorted(scores, key=lambda x: x[1], reverse=True)[:3]

    print("\nTop 3 most relevant documents:")
    for name, score in top3:
        print(f"  - {name} (score={score:.2f})")

    # Combine contexts of top 3 documents
    combined_context = "\n\n---\n\n".join(docs[name]() for name, _ in top3)

    # Ask model using the combined context
    response = wema_analyst.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful banking assistant."},
            {"role": "user", "content": f"""
            Using the following combined context from the top three retrieved documents:

            {combined_context}

            Now answer the question clearly and concisely:

            Question: {question}
            """}
        ]
    )

    print("\nAnswer:")
    print(response.choices[0].message.content)
