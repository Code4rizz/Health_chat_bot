import os
import re
import json
import streamlit as st
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()
st.set_page_config(page_title="Health Agent", page_icon="🏥")
st.title("🏥 Health Agent")


# =========================
# 🔹 Conversational Memory
# =========================
def get_chat_memory(max_messages=10):
    if "messages" not in st.session_state:
        return ""

    history = st.session_state.messages[-max_messages:]
    memory = []

    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        memory.append(f"{role}: {msg['content']}")

    return "\n".join(memory)


# =========================
# 🔹 Vector DB
# =========================
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "health_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

db = load_vector_db()


# =========================
# 🔹 LLM
# =========================
groq_model = Groq(
    id="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)


# =========================
# 🔹 Tools
# =========================
def bmi_tool(weight_kg: float, height_cm: float) -> dict:
    if weight_kg <= 0 or height_cm <= 0:
        return {"error": "Weight and height must be positive numbers."}

    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal (healthy range)"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return {"bmi": round(bmi, 1), "category": category}


# =========================
# 🔹 Agents
# =========================
classifier_agent = Agent(
    model=groq_model,
    instructions="""
Classify the query into EXACTLY ONE category.

GENERAL → non-health questions
HEALTH  → health-related questions

Respond ONLY in JSON:
{"route": "GENERAL"}
"""
)

general_agent = Agent(
    model=groq_model,
    instructions="""
You are a helpful general-purpose assistant.

Rules:
- Answer questions that are NOT related to health or medicine.
- Provide clear, accurate, and concise explanations.
- Do NOT provide medical advice.
"""
)

health_agent = Agent(
    model=groq_model,
    tools=[bmi_tool],
    instructions="""
You are a professional doctor.

Rules:
- Provide general medical guidance only.
- Do NOT diagnose diseases.
- Do NOT give emergency instructions.
"""
)


# =========================
# 🔹 Helpers
# =========================
def extract_weight_height_cm(text):
    nums = [float(x) for x in re.findall(r"\d+\.?\d*", text)]
    if len(nums) < 2:
        return None, None

    weight, height = nums[:2]
    if height < 50:
        return None, None

    return weight, height


def retrieve_context(query, max_distance=1.0):
    results = db.similarity_search_with_score(query, k=1)
    if not results:
        return None, None

    best_doc, score = results[0]
    if score > max_distance:
        return None, None

    source = best_doc.metadata.get("source")
    file_docs = [
        d for d in db.docstore._dict.values()
        if d.metadata.get("source") == source
    ]

    context = "\n\n".join(d.page_content for d in file_docs)
    return context, source


# =========================
# 🔹 Health Handler (WITH MEMORY)
# =========================
def handle_health(query):
    memory = get_chat_memory()
    lower = query.lower()

    if "bmi" in lower and len(re.findall(r"\d+\.?\d*", query)) == 1:
        prompt = f"{memory}\n\n{query}"
        answer = health_agent.run(prompt).content
        return "[AGENT: HEALTH]\n[DOCS USED: NO]\n\n" + answer

    if any(x in lower for x in ["calculate bmi", "based on my bmi", "am i healthy"]):
        weight, height = extract_weight_height_cm(query)

        if not weight or not height:
            return (
                "[AGENT: HEALTH]\n[DOCS USED: NO]\n\n"
                "Please provide your weight (kg) and height (cm)."
            )

        result = bmi_tool(weight, height)
        return (
            "[AGENT: BMI]\n[DOCS USED: NO]\n\n"
            f"Your BMI is **{result['bmi']}** ({result['category']})."
        )

    context, source = retrieve_context(query)
    if context:
        prompt = f"""
Conversation so far:
{memory}

Medical reference:
{context}

User question:
{query}
"""
        answer = health_agent.run(prompt).content
        return (
            "[AGENT: HEALTH]\n"
            "[DOCS USED: YES]\n"
            f"[SOURCE: {source}]\n\n"
            + answer
        )

    prompt = f"{memory}\n\n{query}"
    answer = health_agent.run(prompt).content
    return "[AGENT: HEALTH]\n[DOCS USED: NO]\n\n" + answer


# =========================
# 🔹 Router (WITH MEMORY)
# =========================
def route_query(user_input):
    memory = get_chat_memory()

    result = classifier_agent.run(
        f"{memory}\n\n{user_input}"
    ).content

    try:
        route = json.loads(result)["route"]
    except Exception:
        route = "GENERAL"

    if route == "GENERAL":
        answer = general_agent.run(
            f"{memory}\n\n{user_input}"
        ).content
        return "[AGENT: GENERAL]\n[DOCS USED: NO]\n\n" + answer

    return handle_health(user_input)


# =========================
# 🔹 Streamlit UI
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        response = route_query(user_input)
        st.write(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
