This project was engineered to move beyond basic RAG "wrappers" and address the core challenges of enterprise-grade document intelligence: Data Integrity, Hallucination Control, and User Agency.

Hybrid Memory Management: By utilizing LangChain’s Buffer Window Memory, the system maintains conversational context without allowing the LLM to drift into "hallucination loops" during long sessions.

Vector Optimized Retrieval: Instead of basic top-k similarity, this architecture implements Semantic Chunking with a focus on overlap-integrity. This ensures that metadata—like table headers or section titles—is never severed from its context during the embedding process.

The "Editable" Innovation: Most RAG systems are "Read-Only." This pipeline introduces an Editable PDF Interface, allowing a human-in-the-loop (HITL) to correct or annotate extracted data before it is committed to downstream databases—a critical requirement for finance and legal compliance.

Streamlit as an Enterprise Dashboard: Streamlit was chosen not just for UI, but for its ability to handle asynchronous state management, providing a low-latency experience for end-users interacting with large (100MB+) vector stores.
