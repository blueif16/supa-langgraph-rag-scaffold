from typing import BinaryIO

class DocumentLoader:
    """文档加载器基类"""

    @staticmethod
    def load_text(file: BinaryIO) -> str:
        """加载纯文本"""
        return file.read().decode('utf-8')

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """文本分块"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks
