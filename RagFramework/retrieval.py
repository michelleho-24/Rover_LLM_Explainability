import os
import hashlib
import json
from typing import List, Dict
import numpy as np


class Retriever:

    def __init__(self, sim_path: str = 'data/sim_context.txt', tree_path: str = 'data/test_tree.txt', mission_path: str = 'data/mission_summary.txt', mission_doc_path: str = 'data/mission_doc.pdf'):
        self.sim_path = sim_path
        self.tree_path = tree_path
        self.mission_path = mission_path
        self.mission_doc_path = mission_doc_path

        self.mission_chunks = []
        self.chunk_embeddings = []
        self.load_mission_doc()

    def pdf_hash(self) -> str:
        h = hashlib.md5()
        with open(self.mission_doc_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()

    def cache_path(self) -> str:
        return self.mission_doc_path + ".embeddings.json"

    def load_mission_doc(self):
        if not os.path.exists(self.mission_doc_path):
            return

        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(self.mission_doc_path)
            doc_text = ""
            for page in reader.pages:
                doc_text += page.extract_text()

            self.mission_chunks = self.chunk_text(doc_text, max_chars=2000)

            pdf_hash = self.pdf_hash()
            cache_file = self.cache_path()
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    cache = json.load(f)
                if cache.get("hash") == pdf_hash and len(cache.get("embeddings", [])) == len(self.mission_chunks):
                    self.chunk_embeddings = cache["embeddings"]
                    print(f"[Info] Loaded cached embeddings for {len(self.mission_chunks)} chunks.")
                    return

            from RagFramework.utils import config
            import openai

            print(f"[Info] Computing embeddings for {len(self.mission_chunks)} mission document chunks...")
            client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            for chunk in self.mission_chunks:
                if chunk.strip():
                    response = client.embeddings.create(
                        input=[chunk.replace("\n", " ").strip()],
                        model="text-embedding-3-small"
                    )
                    self.chunk_embeddings.append(response.data[0].embedding)
                else:
                    self.chunk_embeddings.append([0.0] * 1536)

            with open(cache_file, "w") as f:
                json.dump({"hash": pdf_hash, "embeddings": self.chunk_embeddings}, f)
            print(f"[Info] Mission document ready (embeddings cached).")

        except Exception as e:
            print(f"[Warning] Could not load mission PDF: {e}")
            self.mission_chunks = []
            self.chunk_embeddings = []

    def chunk_text(self, text, max_chars=2000):
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_length += len(word) + 1
            if current_length > max_chars:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def get_relevant_mission_chunks(self, query: str, top_k: int = 3) -> str:
        if not self.mission_chunks or not self.chunk_embeddings:
            return ""

        try:
            from RagFramework.utils import config
            import openai

            client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

            response = client.embeddings.create(
                input=[query.replace("\n", " ").strip()],
                model="text-embedding-3-small"
            )
            query_embedding = response.data[0].embedding

            similarities = []
            for i, chunk_emb in enumerate(self.chunk_embeddings):
                similarity = np.dot(query_embedding, chunk_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb)
                )
                similarities.append((similarity, i))

            similarities.sort(reverse=True, key=lambda x: x[0])
            top_chunks = [self.mission_chunks[idx] for _, idx in similarities[:top_k]]

            return "\n\n".join(top_chunks)

        except Exception as e:
            print(f"[Warning] Semantic search failed: {e}")
            return ""

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        artifacts: List[Dict] = []

        if os.path.exists(self.sim_path):
            try:
                with open(self.sim_path, 'r') as f:
                    sim_text = f.read()
                artifacts.append({
                    'label': 'Sim-Context',
                    'content': sim_text,
                    'source': os.path.basename(self.sim_path)
                })
            except Exception:
                pass

        if os.path.exists(self.mission_path):
            try:
                with open(self.mission_path, 'r') as f:
                    mission_text = f.read()
                artifacts.append({
                    'label': 'Mission-Summary',
                    'content': mission_text,
                    'source': self.mission_path
                })
            except Exception:
                pass

        relevant_mission_text = self.get_relevant_mission_chunks(query, top_k=3)
        if relevant_mission_text:
            artifacts.append({
                'label': 'Mission-Document',
                'content': relevant_mission_text,
                'source': self.mission_doc_path
            })

        if not artifacts:
            artifacts = [{'label': 'Fallback', 'content': 'No local RAG artifacts found.', 'source': ''}]

        return artifacts[:k]
