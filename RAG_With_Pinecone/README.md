# Retrieval-Augmented Generation (RAG) with Pinecone and OpenAI

## 📌 Project Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system using **Pinecone** as the vector database and **OpenAI's GPT** for generating responses. The model enhances LLM capabilities by retrieving relevant information from a dataset before generating context-aware responses. 

## 🎯 Key Features
- **Vector Search with Pinecone**: Efficient semantic search over a dataset.
- **OpenAI Embeddings**: Uses `text-embedding-ada-002` for creating vector representations.
- **Contextual Querying**: Retrieves relevant documents to improve LLM responses.
- **Dynamic Prompt Construction**: Builds informative prompts using the retrieved context.
- **Scalable RAG Pipeline**: Can process large datasets with Pinecone indexing.

## 🚀 Technologies Used
- **Python** 🐍
- **Pinecone** 🔍 (Vector Database)
- **OpenAI API** 🤖 (LLM and Embeddings)
- **Pandas** 📊 (Data Processing)
- **TQDM** ⏳ (Progress Bar)

## 📂 Dataset
The dataset used consists of **Wikipedia articles** with metadata. It is loaded from a CSV file containing **text embeddings and article metadata**.

## 🔧 Setup and Installation
```bash
# Clone the repository
git clone https://github.com/your-repo-url.git
cd your-repo-folder

# Install dependencies
pip install openai pinecone-client pandas tqdm datasets
```

## ⚙️ Configuration
1. **Set up Pinecone**:
   - Get your Pinecone API key.
   - Initialize the Pinecone index with `dimension=1536` (matching OpenAI embeddings).

2. **Set up OpenAI**:
   - Get your OpenAI API key.
   - Use `text-embedding-ada-002` for embedding generation.

## 📌 Usage
### 1️⃣ Load the Dataset
```python
import pandas as pd
df = pd.read_csv('./data/wiki.csv', nrows=500)
```

### 2️⃣ Generate and Store Embeddings in Pinecone
```python
from pinecone import Pinecone
import ast

pinecone = Pinecone(api_key='YOUR_PINECONE_API_KEY')
index = pinecone.Index('your-index-name')

prepped = []
for i, row in df.iterrows():
    meta = ast.literal_eval(row['metadata'])
    prepped.append({'id': row['id'], 'values': ast.literal_eval(row['values']), 'metadata': meta})
index.upsert(prepped)
```

### 3️⃣ Query the RAG System
```python
query = "What is the Berlin Wall?"
embed = get_embeddings([query])
res = index.query(vector=embed.data[0].embedding, top_k=3, include_metadata=True)
```

### 4️⃣ Build the Prompt
```python
contexts = [x['metadata']['text'] for x in res['matches']]
prompt = "\n\n---\n\n".join(contexts) + f"\n\nQuestion: {query}\nAnswer:"
```

### 5️⃣ Generate Answer with OpenAI
```python
res = openai_client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    temperature=0,
    max_tokens=636
)
print(res.choices[0].text)
```

## 📊 Results
- Queries are answered with **contextually relevant information**.
- Improved **accuracy and relevance** compared to standard LLM responses.
- Can handle **large-scale retrieval and indexing**.

## 🔥 Future Improvements
- Expand to **multi-source retrieval** (e.g., web, PDFs, APIs).
- Fine-tune retrieval ranking for **better accuracy**.
- Implement **RAG evaluation metrics** for response quality.

## 📜 License
This project is licensed under the **MIT License**.

## ✉️ Contact
For any questions or collaborations, reach out at: [katreddisrisaidurga@gmail.com](mailto:katreddisrisaidurga@gmail.com)
