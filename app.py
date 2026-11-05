"""Streamlit demo: streamlit run app.py"""
import streamlit as st

from src.data import load_jsonl
from src.generate import generate
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Reranker

st.set_page_config(page_title="FactCheck-RAG", page_icon="🔎")
st.title("FactCheck-RAG")
st.caption("Verifica un'affermazione in italiano o spagnolo con fact-check già pubblicati.")


@st.cache_resource
def load():
    fcs, _ = load_jsonl("data/sample_factchecks.jsonl")
    bm25 = BM25Retriever(fcs)
    return HybridRetriever([bm25, DenseRetriever(fcs)]), Reranker()


retriever, reranker = load()
claim = st.text_area("Affermazione da verificare")
if st.button("Verifica") and claim.strip():
    with st.spinner("Cerco fact-check..."):
        docs = reranker.rerank(claim, retriever.search(claim, k=20), k=5)
    with st.spinner("Scrivo la risposta..."):
        st.markdown(generate(claim, docs))
    st.subheader("Fonti")
    for i, (fc, score) in enumerate(docs, 1):
        st.markdown(f"**[{i}]** {fc.claim} — *{fc.verdict}* ({fc.lang}) {fc.url}")
