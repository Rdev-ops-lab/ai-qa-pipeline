"""
Factual accuracy scorer using retrieval grounding.
Full mode: LangChain + FAISS vector search against reference context.
Mock mode: deterministic stub for testing and CI.
"""
import os


class FactualScorer:
    def __init__(self, mock: bool = False):
        self.mock = mock

    def score(self, response_text: str, reference_context: str) -> float:
        if self.mock:
            return self._mock_score(response_text, reference_context)
        return self._retrieval_score(response_text, reference_context)

    def _mock_score(self, response_text: str, reference_context: str) -> float:
        if not reference_context:
            return 0.6
        overlap = len(set(response_text.lower().split()) & set(reference_context.lower().split()))
        return min(1.0, 0.4 + overlap * 0.03)

    def _retrieval_score(self, response_text: str, reference_context: str) -> float:
        # Full implementation: LangChain + FAISS retrieval grounding
        # Requires OPENAI_API_KEY
        try:
            from langchain_openai import OpenAIEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
            chunks = splitter.split_text(reference_context or response_text)
            embeddings = OpenAIEmbeddings()
            store = FAISS.from_texts(chunks, embeddings)
            docs = store.similarity_search_with_score(response_text, k=3)
            if not docs:
                return 0.5
            scores = [1 - min(dist, 1.0) for _, dist in docs]
            return round(sum(scores) / len(scores), 4)
        except Exception:
            return self._mock_score(response_text, reference_context)
