import re
import tiktoken


class SemanticChunker:
    def __init__(self, max_tokens=200, overlap=20):
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.encoder = tiktoken.get_encoding("gpt2")

    def split_by_tokens(self, text):
        tokens = self.encoder.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk = self.encoder.decode(chunk_tokens)
            chunks.append(chunk.strip())
            start += self.max_tokens - self.overlap
        return chunks

    def chunk(self, text, metadata=None):
        chunks = self.split_by_tokens(text)
        return [
            {
                "title": f"Chunk {i+1}",
                "text": chunk,
                "page": metadata.get("page", 0) if metadata else 0,
            }
            for i, chunk in enumerate(chunks)
        ]
