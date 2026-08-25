import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

# Load environment variables
load_dotenv()

# Streamlit Page Setup
st.set_page_config(page_title="CrewAI Cloud Studio", page_icon="☁️", layout="wide")
st.title("☁️ CrewAI Cloud Agent Studio")
st.caption("Run multi-agent workflows using OpenRouter cloud models.")

# Retrieve OpenRouter API Key (Supports local .env and Streamlit Cloud Secrets)
openrouter_key = os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENROUTER_API_KEY", "")

# Sidebar Configuration
st.sidebar.header("Model Selection")
openrouter_model = st.sidebar.selectbox(
    "Select Cloud Model:",
    [
        "anthropic/claude-3.5-sonnet",
        "deepseek/deepseek-chat",
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemini-flash-1.5"
    ]
)

# Internal Safeguard Settings (Hidden from UI)
MAX_ITERATIONS = 3
MAX_EXEC_TIME = 120

# API Key Check
if not openrouter_key:
    st.error("⚠️ Missing OPENROUTER_API_KEY! Please set it in your `.env` file locally or in Streamlit Secrets.")
    st.stop()

# Initialize OpenRouter LLM Instance using OpenAI-compatible specification
llm = LLM(
    model=openrouter_model,
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key
)

# Main User Inputs
st.subheader("Task Definition")
topic = st.text_input("Research Topic:", "")
task_instructions = st.text_area(
    "Specific Instructions:",
    "Provide a detailed summary and highlight 3 key insights."
)

# Run Crew Execution
if st.button("🚀 Run Cloud Crew", type="primary"):
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        with st.spinner("Agents executing task via OpenRouter Cloud..."):
            try:
                # 1. Define Agent with Internal Safeguards
                researcher = Agent(
                    role="Cloud Research Specialist",
                    goal=f"Analyze and summarize insights on {topic}",
                    backstory="You are a precise technical researcher focused on delivering direct, high-value insights.",
                    llm=llm,
                    max_iter=MAX_ITERATIONS,          # Hard limit on agent loop retries
                    max_execution_time=MAX_EXEC_TIME,  # Hard timeout in seconds
                    verbose=True
                )

                # 2. Define Task
                research_task = Task(
                    description=f"Topic: {topic}\nInstructions: {task_instructions}",
                    expected_output="A clean, structured Markdown report with clear headings and bullet points.",
                    agent=researcher
                )

                # 3. Assemble Crew with Constraints
                crew = Crew(
                    agents=[researcher],
                    tasks=[research_task],
                    max_iter=MAX_ITERATIONS,
                    memory=False  # Disables local vector store/embeddings
                )

                # 4. Execute
                result = crew.kickoff()

                # 5. Render Output
                st.success("Execution Completed Successfully!")
                st.markdown("---")
                st.markdown("### 📝 Generated Report")
                st.markdown(result.raw)

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
