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
        answer, docs = service.verify(claim, k=5)
    st.markdown(answer)
    st.subheader("Fonti")
    for i, (fc, score) in enumerate(docs, 1):
        st.markdown(f"**[{i}]** {fc.claim} — *{fc.verdict}* ({fc.lang}) {fc.url}")
