"""Streamlit demo: streamlit run app.py"""
import streamlit as st

from src.service import FactCheckService

st.set_page_config(page_title="FactCheck-RAG", page_icon="🔎")
st.title("FactCheck-RAG")
st.caption("Verifica un'affermazione in italiano o spagnolo con fact-check già pubblicati.")


@st.cache_resource
def load():
    return FactCheckService.from_env()


service = load()
claim = st.text_area("Affermazione da verificare")
if st.button("Verifica") and claim.strip():
    with st.spinner("Cerco i fact-check e scrivo la risposta..."):
        try:
            answer, docs = service.verify(claim, k=5)
            st.markdown(answer)
        except Exception:
            # No LLM available (e.g. Ollama not running): show the retrieved sources only.
            docs = service.search(claim, k=5)
            st.info("LLM non disponibile: mostro solo i fact-check recuperati (retrieval).")
    st.subheader("Fonti")
    for i, (fc, score) in enumerate(docs, 1):
        st.markdown(f"**[{i}]** {fc.claim} — *{fc.verdict}* ({fc.lang}) · score {score:.3f}")
