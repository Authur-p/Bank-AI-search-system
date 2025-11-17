from openai import AzureOpenAI
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def load_file(path):
    with open(path, 'r') as file:
        document_text =  file.read()
    return document_text

wema_analyst = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint = os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("API_KEY"))


docs = {
    'bank_policy': lambda: load_file('bank_policy.txt'),
    'loan_policy': lambda: load_file('loans_policy.txt'),
    'faq_txt': lambda: load_file('Bank_FAQ.txt')
}



embeddings = {}

for key, value in docs.items():
    res = wema_analyst.embeddings.create(input=value(), model="text-embedding-3-small")
    embeddings[key] = np.array(res.data[0].embedding)


question = "What is th emaximum amount for user loans"
q_embed = np.array(wema_analyst.embeddings.create(input=question, model="text-embedding-3-small").data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))   

best_doc = None
best_score = -1

for name, emb in embeddings.items():
    score = cosine_similarity(q_embed, emb)
    if score > best_score:
        best_doc = name
        best_score = score

print(f"Best match: {best_doc} (score={best_score:.2f})")

context = docs[best_doc]()
response = wema_analyst.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful banking assistant."},
        {"role": "user", "content": f"Usint this document:{context}\n\nQuestion: {question}"}
    ]
)
print(response.choices[0].message.content)