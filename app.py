import streamlit as st
from utils.rag import answer

st.set_page_config(page_title="PadCare GPT (Public RAG)", layout="wide")

st.title("PadCare GPT")
st.caption("Answers questions about PadCare Labs using only publicly available sources (RAG + Mistral via Ollama).")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Output type")
    intent = st.selectbox(
        "Choose format",
        options=["qa", "simple", "investor_pitch", "linkedin_post", "bullets"],
        index=0,
        help="Same data, different output formatting."
    )
    k = st.slider("Top-K sources", min_value=3, max_value=10, value=5, step=1)

    st.subheader("Demo prompts")
    demo_prompts = [
        "What problem does PadCare solve?",
        "Explain PadCare’s process in simple words.",
        "Create a short investor pitch for PadCare.",
        "Write a LinkedIn post about PadCare’s impact.",
        "Summarize PadCare’s work in 5 bullet points."
    ]
    demo = st.radio("Pick one", demo_prompts, index=0)
    if st.button("Use demo prompt"):
        st.session_state["q"] = demo

with col1:
    q = st.text_area("Ask a question", value=st.session_state.get("q", ""), height=140, placeholder="Type your question about PadCare...")

    show_sources = st.checkbox("Show retrieved sources/chunks", value=True)

    if st.button("Answer", type="primary"):
        if not q.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving public sources and generating answer..."):
                out = answer(q.strip(), intent=intent, k=k)

            st.subheader("Answer")
            st.write(out["answer"])

            if show_sources:
                st.subheader("Retrieved context (evidence)")
                for i, item in enumerate(out["retrieved"], 1):
                    m = item["metadata"]
                    with st.expander(f"{i}. {m['source_name']} — {m['doc_title']}"):
                        st.write(item["text"])
                        st.caption(f"Source: {m['source_url']}")