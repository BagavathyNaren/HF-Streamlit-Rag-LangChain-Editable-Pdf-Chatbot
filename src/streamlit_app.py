import os
import re
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# Setup
# -----------------------------
load_dotenv()

def find_pdf_dir():
    """Starts at script location and looks up for 'research_papers' folder."""
    current_path = os.path.abspath(__file__)
    # Go up a maximum of 3 levels to find the folder
    for _ in range(3):
        current_path = os.path.dirname(current_path)
        potential_path = os.path.join(current_path, "research_papers")
        if os.path.exists(potential_path):
            return potential_path
    return None

PDF_DIR = find_pdf_dir()

# Validation and Debugging
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing")
    st.stop()

if PDF_DIR is None:
    st.error(f"FATAL: Could not find 'research_papers' folder. Script is at: {os.path.abspath(__file__)}")
    st.info(f"Current Working Directory: {os.getcwd()}")
    st.stop()
else:
    # Optional success message - can be commented out for cleaner UI
    st.sidebar.success(f"Connected to: {PDF_DIR}")

# -----------------------------
# BULLETPROOF PDF CLEANER
# -----------------------------
def clean_pdf_text(text: str) -> str:
    if not text:
        return ""

    # Fix OCR letter spacing FIRST
    text = re.sub(
        r"(?:\b[A-Za-z]\s){2,}[A-Za-z]\b",
        lambda m: m.group(0).replace(" ", ""),
        text
    )

    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()

# -----------------------------
# CLEAN EXCERPT (FIXED)
# -----------------------------
def make_excerpt(text: str, max_len: int = 400) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    cut = text.rfind(" ", 0, max_len)
    return text[:cut] + "..."

# -----------------------------
# MANUAL COSINE NORMALIZATION
# -----------------------------
class NormalizedOpenAIEmbeddings(OpenAIEmbeddings):
    def embed_documents(self, texts):
        vectors = super().embed_documents(texts)
        return [self._normalize(v) for v in vectors]

    def embed_query(self, text):
        v = super().embed_query(text)
        return self._normalize(v)

    def _normalize(self, v):
        v = np.array(v)
        norm = np.linalg.norm(v)
        return (v / norm).tolist() if norm > 0 else v.tolist()

embeddings = NormalizedOpenAIEmbeddings(
    model="text-embedding-3-small"
)

# -----------------------------
# LLM & PROMPT
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

prompt = ChatPromptTemplate.from_template(
"""
Answer using bullet points ONLY.

Rules:
- Each bullet MUST contain exactly ONE factual claim.
- Each bullet MUST end with a citation like [Source X].
- Use ONLY the sources provided.
- If the answer is not present, say exactly: "I don't know."

Sources:
{context}

Question:
{question}
"""
)

# -----------------------------
# Helpers
# -----------------------------
def extract_cited_sources(answer: str):
    return set(map(int, re.findall(r"Source (\d+)", answer)))

def extract_entities(text: str):
    return set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text))

def rerank(query, docs_with_scores):
    query_tokens = set(query.lower().split())
    query_entities = extract_entities(query)
    reranked = []

    for doc, score in docs_with_scores:
        doc_tokens = set(doc.page_content.lower().split())
        doc_entities = extract_entities(doc.page_content)

        token_overlap = len(query_tokens & doc_tokens)
        entity_overlap = len(query_entities & doc_entities)

        rerank_score = (
            score * 0.6 +
            token_overlap * 0.25 +
            entity_overlap * 0.15
        )
        reranked.append((doc, score, rerank_score))

    reranked.sort(key=lambda x: x[2], reverse=True)
    return reranked

# -----------------------------
# Vector Store Creation
# -----------------------------
def create_vector_store():
    if "vectorstore" in st.session_state:
        return

    documents = []
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        st.error(f"No PDF files found in {PDF_DIR}")
        st.stop()

    for file in pdf_files:
        loader = PyPDFLoader(os.path.join(PDF_DIR, file))
        pages = loader.load()

        for p in pages:
            p.page_content = clean_pdf_text(p.page_content)
            # ✅ FIX: Convert page index to human page number
            p.metadata["page"] = p.metadata.get("page", 0) + 1

            if len(p.page_content) > 50:
                documents.append(p)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        st.error("No valid text chunks created.")
        st.stop()

    st.session_state.vectorstore = FAISS.from_documents(
        chunks,
        embeddings,
        distance_strategy="IP"
    )
    st.success(f"Indexed {len(chunks)} chunks from {len(pdf_files)} files.")

# -----------------------------
# UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("RAG PDF Q&A with OpenAI (Citations + Confidence)")

query = st.text_input("Ask a question from the documents")

if st.button("Create Document Index"):
    create_vector_store()

# -----------------------------
# RAG FLOW
# -----------------------------
if query:
    if "vectorstore" not in st.session_state:
        st.error("Please create the document index first.")
        st.stop()

    vectorstore = st.session_state.vectorstore
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=8)

    if not docs_with_scores:
        st.write("I don't know.")
        st.stop()

    reranked = rerank(query, docs_with_scores)[:4]
    best_score = reranked[0][1]

    if best_score < 0.65:
        st.write("I don't know.")
        st.stop()

    context = ""
    docs = []

    for i, (doc, score, rerank_score) in enumerate(reranked, start=1):
        context += f"[Source {i}]\n{doc.page_content}\n\n"
        docs.append(doc)

    response = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": query
    })

    lines = [l.strip() for l in response.splitlines() if l.strip()]
    bullets = [l for l in lines if l.startswith("-")]

    if not bullets:
        st.error("Answer must be bullet-point only.")
        st.stop()

    for line in bullets:
        if "[Source" not in line:
            st.error("Each bullet must contain a citation.")
            st.stop()

    st.subheader("Answer")
    st.write("\n".join(bullets))

    cited_sources = extract_cited_sources(response)

    st.subheader("Sources")
    for i, doc in enumerate(docs, start=1):
        if i not in cited_sources:
            continue

        st.markdown(
            f"""
            **Source {i}**
            - File: `{os.path.basename(doc.metadata.get("source", "Unknown"))}`
            - Page: `{doc.metadata.get("page")}`
            - Excerpt:
            > {make_excerpt(doc.page_content)}
            """
        )