🏥 Health Chat Bot (RAG + BMI Tool)

An AI-powered health assistant built with Streamlit that intelligently routes user queries between a general assistant and a health-focused agent, supports retrieval-augmented generation (RAG) using FAISS, and performs accurate BMI calculations using a deterministic tool.

This project is designed to be safe, modular, debuggable, and deployable on Streamlit Community Cloud.

✨ Features
🧠 Intelligent Routing

Automatically classifies queries into:

GENERAL → non-health questions

HEALTH → health-related questions

🩺 Health Agent

Provides general medical guidance only

Uses documents only when relevant

Interprets BMI if already provided

Calculates BMI only when necessary

🧮 BMI Tool (Deterministic)

Uses exact math (no hallucination)

Requires:

Weight in kg

Height in cm

Returns BMI value + category

📚 Retrieval-Augmented Generation (RAG)

Uses FAISS vector database

Injects only highly relevant documents

Displays source document for debugging

🧪 Debug-Friendly Output

Each response shows:

[AGENT: ...]
[DOCS USED: YES / NO]
[SOURCE: ...] (if applicable)

🧱 Architecture Overview
User Query
   ↓
Classifier Agent
   ↓
 ┌───────────────┐
 │ GENERAL Agent │ → General knowledge answers
 └───────────────┘
        OR
 ┌───────────────────────────────┐
 │ Health Pipeline               │
 │  • BMI interpretation         │
 │  • BMI calculation (tool)     │
 │  • RAG document injection     │
 │  • Plain health advice        │
 └───────────────────────────────┘

📂 Project Structure
health_chat_bot/
├── app.py                  # Main Streamlit app
├── ingest.py               # Builds FAISS vector index (one-time)
├── data/                   # Health documents (.txt)
│   ├── water.txt
│   ├── sleep.txt
│   ├── exercise.txt
│   └── nutrition.txt
├── health_index/            # Generated FAISS index
│   ├── index.faiss
│   └── index.pkl
├── requirements.txt
├── README.md
└── .env                    # Local only (not committed)

⚙️ Installation (Local)
1️⃣ Clone the repository
git clone https://github.com/Code4rizz/Health_chat_bot.git
cd Health_chat_bot

2️⃣ Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set environment variable

Create a .env file:

GROQ_API_KEY=your_groq_api_key_here

📚 Build Vector Index (One-Time)

If health_index/ does not exist:

python ingest.py


This:

Reads documents from data/

Splits into chunks

Creates embeddings

Saves FAISS index to health_index/

▶️ Run the App
streamlit run app.py


Open:

http://localhost:8501

🌐 Streamlit Cloud Deployment

Push code to GitHub

Ensure requirements.txt exists

Log into Streamlit Cloud with the same GitHub account

New App → From GitHub

Select:

Repository: Code4rizz/Health_chat_bot

Branch: main

File: app.py

Add Secrets:

GROQ_API_KEY = your_groq_api_key_here

🧪 Example Queries
General
prime minister of india
what is machine learning

Health (No Docs)
is walking good for health

Health + Docs (RAG)
how much water should i drink daily
how many hours should i sleep

BMI Interpretation
my bmi is 24.4 am i healthy

BMI Calculation
calculate bmi for 70 kg 170 cm
am i healthy 65 168

⚠️ Disclaimer

This application provides general health information only.
It does not diagnose diseases and is not a replacement for professional medical advice.

🚀 Future Improvements

Unit support (lbs/inches)

Confidence scores for RAG

Multiple document sources

UI toggle for debug mode

Automated tests

Modular codebase
