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

groq_model = Groq(
    id="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

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

classifier_agent = Agent(
    model=groq_model,
    instructions="""
Classify the query into EXACTLY ONE category.
GENERAL → non-health questions
HEALTH → health-related questions
Respond ONLY in JSON: {"route": "GENERAL"}
"""
)

general_agent = Agent(
    model=groq_model,
    instructions="""
You are a helpful general-purpose assistant.

Rules:
- Answer questions that are NOT related to health or medicine.
- Provide clear, accurate, and concise explanations.
- If the question is factual, give a direct answer first, then brief context if helpful.
- If the question is opinion-based, present a balanced and neutral view.
- Do NOT provide medical, health, or diagnostic advice.
- If a question seems health-related, answer briefly at a high level and avoid giving medical guidance.

Style:
- Be polite, calm, and confident.
- Avoid unnecessary disclaimers.
- Keep responses easy to understand.
"""
)

health_agent = Agent(
    model=groq_model,
    tools=[bmi_tool],
    instructions="""
You are a professional doctor.

General rules:
- Provide general medical guidance only.
- Do NOT diagnose diseases.
- Do NOT give emergency instructions.

CRITICAL TOOL RULES (IMPORTANT):
- You MUST NOT call the bmi_tool unless BMI calculation is explicitly required.
- ONLY call bmi_tool when:
  (a) the user explicitly asks to calculate BMI, OR
  (b) the user asks for a health judgement that requires BMI AND weight (kg) and height (cm) are provided.
- NEVER call bmi_tool for hydration, sleep, exercise, diet, or any general health advice.
- If BMI is already provided, DO NOT call bmi_tool.

BMI rules:
- BMI calculation requires weight in kilograms (kg) and height in centimeters (cm).
- If required values are missing or unclear, ask the user to provide them. Do NOT guess.

Context rules:
- You may receive reference information.
- Use it ONLY if relevant.
- If no reference is provided, answer normally.

Output rules:
- Do NOT mention tools, function calls, or internal steps.
- Respond with plain natural language only.

Be clear, practical, and concise.
"""
)

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

def get_conversation_history():
    """Get the last 5 user prompts for context"""
    history = []
    for msg in st.session_state.messages[-10:]:  # Get last 10 messages (5 pairs)
        if msg["role"] == "user":
            history.append(msg["content"])
    return history[-5:]  # Return last 5 user prompts

def build_prompt_with_history(query):
    """Build prompt with conversation history"""
    history = get_conversation_history()
    if not history:
        return query
    
    history_text = "Previous conversation:\n"
    for i, msg in enumerate(history, 1):
        history_text += f"{i}. {msg}\n"
    
    return f"{history_text}\nCurrent question: {query}"

def handle_health(query):
    lower = query.lower()
    
    if "bmi" in lower and len(re.findall(r"\d+\.?\d*", query)) == 1:
        prompt = build_prompt_with_history(query)
        answer = health_agent.run(prompt).content
        return "[AGENT: HEALTH]\n[DOCS USED: NO]\n\n" + answer
    
    if any(x in lower for x in ["calculate bmi", "based on my bmi", "am i healthy"]):
        weight, height = extract_weight_height_cm(query)
        if not weight or not height:
            return (
                "[AGENT: HEALTH]\n[DOCS USED: NO]\n\n"
                "if your are trying to calculate your BMI, "
                "Please provide your weight in kilograms (kg) "
                "and height in centimeters (cm)."
            )
        result = bmi_tool(weight, height)
        if "error" in result:
            return "[AGENT: BMI]\n[DOCS USED: NO]\n\n" + result["error"]
        return (
            "[AGENT: BMI]\n[DOCS USED: NO]\n\n"
            f"Your BMI is **{result['bmi']}**, which falls in the "
            f"**{result['category']}** category.\n\n"
            "Maintaining balanced nutrition, regular physical activity, "
            "and good sleep habits supports overall health."
        )
    
    context, source = retrieve_context(query)
    prompt = build_prompt_with_history(query)
    
    if context:
        prompt = f"{context}\n\n{prompt}"
        answer = health_agent.run(prompt).content
        return (
            "[AGENT: HEALTH]\n"
            "[DOCS USED: YES]\n"
            f"[SOURCE: {source}]\n\n" + answer
        )
    
    answer = health_agent.run(prompt).content
    return "[AGENT: HEALTH]\n[DOCS USED: NO]\n\n" + answer

def route_query(user_input):
    result = classifier_agent.run(user_input).content
    try:
        route = json.loads(result)["route"]
    except Exception:
        route = "GENERAL"
    
    if route == "GENERAL":
        prompt = build_prompt_with_history(user_input)
        answer = general_agent.run(prompt).content
        return "[AGENT: GENERAL]\n[DOCS USED: NO]\n\n" + answer
    
    return handle_health(user_input)

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