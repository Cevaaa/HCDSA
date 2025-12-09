# -*- coding: utf-8 -*-
"""
独立的 RAG 模块 - 与 Agent 解耦
可单独使用，也可注入到 Agent 中
"""
import json
import hashlib
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RAGDocument:
    """RAG 文档结构"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    doc_id: str = ""
    
    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


class SimpleRAGStore:
    """
    简单的 RAG 存储 - 完全独立，无外部依赖
    
    特性：
    - 支持内存存储和文件持久化
    - 简单的 TF-IDF 相似度（无需额外依赖）
    - 可选接入外部 embedding 模型
    """
    
    def __init__(
        self,
        persist_path: Optional[str] = None,
        similarity_threshold: float = 0.3,
        max_results: int = 5,
    ):
        self.persist_path = Path(persist_path) if persist_path else None
        self.similarity_threshold = similarity_threshold
        self.max_results = max_results
        
        self.documents: Dict[str, RAGDocument] = {}
        self.vocab: Dict[str, int] = {}  # 词汇表
        self.idf: Dict[str, float] = {}  # IDF 值
        
        # 从文件加载
        if self.persist_path and self.persist_path.exists():
            self._load()
    
    def add(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """添加文档到 RAG 存储"""
        doc = RAGDocument(
            content=content,
            metadata=metadata or {},
            doc_id=doc_id or "",
        )
        
        # 去重
        if doc.doc_id in self.documents:
            return doc.doc_id
        
        self.documents[doc.doc_id] = doc
        self._update_index(doc)
        
        # 持久化
        if self.persist_path:
            self._save()
        
        return doc.doc_id
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        if not self.documents:
            return []
        
        top_k = top_k or self.max_results
        query_vec = self._text_to_vector(query)
        
        results = []
        for doc_id, doc in self.documents.items():
            # 元数据过滤
            if filter_metadata:
                if not all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            
            doc_vec = self._text_to_vector(doc.content)
            similarity = self._cosine_similarity(query_vec, doc_vec)
            
            if similarity >= self.similarity_threshold:
                results.append({
                    "doc_id": doc_id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": similarity,
                })
        
        # 按相似度排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def get_or_search(
        self,
        query: str,
        threshold: Optional[float] = None,
    ) -> Optional[str]:
        """
        查询 RAG，如果有高相似度结果则返回，否则返回 None
        这是给 Agent 用的简化接口
        """
        threshold = threshold or self.similarity_threshold
        results = self.search(query, top_k=1)
        
        if results and results[0]["score"] >= threshold:
            return results[0]["content"]
        return None
    
    def _text_to_vector(self, text: str) -> Dict[str, float]:
        """简单的 TF-IDF 向量化"""
        words = self._tokenize(text)
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # TF-IDF
        vector = {}
        for word, count in word_count.items():
            tf = count / len(words) if words else 0
            idf = self.idf.get(word, 1.0)
            vector[word] = tf * idf
        
        return vector
    
    def _cosine_similarity(
        self,
        vec1: Dict[str, float],
        vec2: Dict[str, float],
    ) -> float:
        """计算余弦相似度"""
        all_keys = set(vec1.keys()) | set(vec2.keys())
        
        dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in all_keys)
        norm1 = np.sqrt(sum(v ** 2 for v in vec1.values())) or 1e-10
        norm2 = np.sqrt(sum(v ** 2 for v in vec2.values())) or 1e-10
        
        return dot_product / (norm1 * norm2)
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        import re
        # 转小写，提取单词
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        # 去停用词
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'under', 'again', 'further', 'then', 'once',
                     'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either',
                     'neither', 'not', 'only', 'own', 'same', 'than', 'too',
                     'very', 'just', 'also', 'now', 'here', 'there', 'when',
                     'where', 'why', 'how', 'all', 'each', 'every', 'both',
                     'few', 'more', 'most', 'other', 'some', 'such', 'no',
                     'any', 'this', 'that', 'these', 'those', 'what', 'which',
                     'who', 'whom', 'its', 'it', 'he', 'she', 'they', 'them',
                     'his', 'her', 'their', 'our', 'your', 'my', 'me', 'us',
                     'him', 'we', 'you', 'i'}
        return [w for w in words if w not in stopwords]
    
    def _update_index(self, doc: RAGDocument) -> None:
        """更新索引"""
        words = self._tokenize(doc.content)
        for word in set(words):
            if word not in self.vocab:
                self.vocab[word] = 0
            self.vocab[word] += 1
        
        # 重新计算 IDF
        n_docs = len(self.documents)
        for word, doc_freq in self.vocab.items():
            self.idf[word] = np.log((n_docs + 1) / (doc_freq + 1)) + 1
    
    def _save(self) -> None:
        """持久化到文件"""
        if not self.persist_path:
            return
        
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "documents": {
                doc_id: {
                    "content": doc.content,
                    "metadata": doc.metadata,
                }
                for doc_id, doc in self.documents.items()
            },
            "vocab": self.vocab,
            "idf": self.idf,
        }
        
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load(self) -> None:
        """从文件加载"""
        if not self.persist_path or not self.persist_path.exists():
            return
        
        with open(self.persist_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for doc_id, doc_data in data.get("documents", {}).items():
            self.documents[doc_id] = RAGDocument(
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {}),
                doc_id=doc_id,
            )
        
        self.vocab = data.get("vocab", {})
        self.idf = data.get("idf", {})
    
    def stats(self) -> Dict[str, Any]:
        """返回统计信息"""
        return {
            "total_documents": len(self.documents),
            "vocab_size": len(self.vocab),
            "persist_path": str(self.persist_path) if self.persist_path else None,
        }


# ============ 全局单例（可选使用）============

_GLOBAL_RAG_STORE: Optional[SimpleRAGStore] = None


def get_global_rag_store(
    persist_path: Optional[str] = None,
    **kwargs,
) -> SimpleRAGStore:
    """获取全局 RAG 存储实例"""
    global _GLOBAL_RAG_STORE
    if _GLOBAL_RAG_STORE is None:
        _GLOBAL_RAG_STORE = SimpleRAGStore(persist_path=persist_path, **kwargs)
    return _GLOBAL_RAG_STORE