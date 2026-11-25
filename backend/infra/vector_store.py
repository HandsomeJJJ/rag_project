from langchain_chroma import Chroma
from langchain_core.documents import Document

from core import config


class VectorStoreService:
    def __init__(self, embedding):
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        search_kwargs = {"k": config.retrieval_top_k}
        if config.retrieval_source_filter:
            search_kwargs["filter"] = {"source": config.retrieval_source_filter}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def get_vector_docs(self, query: str, top_k: int, source_filter: str = "") -> list[Document]:
        search_kwargs = {"k": top_k}
        filter_value = source_filter or config.retrieval_source_filter
        if filter_value:
            search_kwargs["filter"] = {"source": filter_value}
        retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
        return retriever.invoke(query)

    def get_all_documents(self, source_filter: str = "") -> list[Document]:
        """从当前 collection 拉取全部文档用于关键词召回。"""
        data = self.vector_store._collection.get(include=["documents", "metadatas"])  # noqa: SLF001
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        items = [
            Document(page_content=doc or "", metadata=meta or {})
            for doc, meta in zip(documents, metadatas)
        ]
        if source_filter:
            items = [
                doc for doc in items
                if str((doc.metadata or {}).get("source", "")) == source_filter
            ]
        return items
