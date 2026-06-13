import streamlit as st
import requests

st.set_page_config(
    page_title="Financial AI Agent",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Financial AI Agent")
st.markdown("Ask any question about Apple, Amazon, Google, Microsoft, or Tesla annual reports.")

# Exemples de questions
st.markdown("### 💡 Example questions")
col1, col2 = st.columns(2)

with col1:
    if st.button("Amazon revenue 2025"):
        st.session_state.question = "What was Amazon total revenue in 2025?"
    if st.button("Apple supply chain risks"):
        st.session_state.question = "What did Apple report say about supply chain risks?"

with col2:
    if st.button("Compare Google vs Microsoft"):
        st.session_state.question = "Compare Microsoft and Google net income in 2024"
    if st.button("Tesla employees 2024"):
        st.session_state.question = "How many employees did Tesla have in 2024?"

st.markdown("---")

# Zone de question
question = st.text_input(
    "Ask your question:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What was Apple net income in 2025?"
)

if st.button("🔍 Ask", type="primary"):
    if question:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    "http://localhost:8000/query",
                    json={"question": question}
                )
                data = response.json()
                
                st.markdown("### 📝 Answer")
                st.write(data["answer"])
                
                st.markdown("---")
                st.caption(f"💰 Token cost: ${data['token_cost']:.6f}")
                
            except Exception as e:
                st.error(f"Error: {str(e)} — Make sure the API is running!")
    else:
        st.warning("Please enter a question!")