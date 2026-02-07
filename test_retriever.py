"""
retriever.py 모듈의 기능을 테스트합니다.
벡터 검색, BM25 검색, 하이브리드 검색, 리랭킹 테스트.
"""

import sys
from pathlib import Path
from datetime import datetime

# app 모듈을 import하기 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from app.retriever import (
    VectorRetriever,
    BM25Retriever,
    HybridRetriever,
    AdvancedRetriever,
    Reranker,
    get_retriever
)
from app.vector_store import ChromaVectorStore, get_vector_store
from app.ollama_client import get_ollama_client


def print_separator(title: str = ""):
    """구분선 출력"""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    else:
        print(f"{'-'*70}\n")


def print_results(results, title="검색 결과"):
    """검색 결과를 보기 좋게 출력"""
    print(f"\n{title}: {len(results)}개")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] ID: {result.id}")
        print(f"    점수: {result.score:.4f}")
        print(f"    내용: {result.content[:100]}...")
        if result.metadata:
            print(f"    메타: {result.metadata}")


def setup_test_data():
    """테스트용 데이터를 벡터 저장소에 추가"""
    print_separator("테스트 데이터 준비")
    
    # 테스트용 컬렉션
    store = ChromaVectorStore(collection_name="test_retriever_collection")
    store.clear()
    
    ollama_client = get_ollama_client()
    
    # 5G 관련 기술 문서
    test_docs = [
        "5G technology operates on various frequency bands including sub-6GHz and mmWave. The 3.5GHz band is commonly used for enhanced mobile broadband services.",
        "Voice over LTE (VoLTE) enables voice calls over 4G LTE networks. It provides better call quality compared to traditional circuit-switched voice.",
        "5G VoLTE interoperability testing is crucial for ensuring seamless voice services across different network generations and vendors.",
        "Long Term Evolution (LTE) is a standard for wireless broadband communication. It forms the foundation for 4G networks worldwide.",
        "The 5G femtocell provides local coverage and capacity enhancement. Small cells are essential for dense urban deployments.",
        "Network slicing in 5G allows operators to create multiple virtual networks on a single physical infrastructure.",
        "Massive MIMO (Multiple Input Multiple Output) is a key technology in 5G that uses many antennas to improve spectral efficiency.",
        "Edge computing in 5G networks reduces latency by processing data closer to the end users.",
        "VoLTE roaming requires careful coordination between mobile operators to ensure service continuity.",
        "5G standalone (SA) architecture provides full 5G capabilities, while non-standalone (NSA) relies on existing LTE infrastructure."
    ]
    
    print(f"문서 {len(test_docs)}개 임베딩 생성 중...")
    
    # 임베딩 생성 및 저장
    ids = []
    embeddings = []
    metadatas = []
    
    for i, doc in enumerate(test_docs):
        doc_id = f"test_doc_{i+1}"
        embedding = ollama_client.embed(doc)
        
        ids.append(doc_id)
        embeddings.append(embedding)
        metadatas.append({
            "source": "test",
            "created_at": datetime.now().isoformat(),
            "doc_num": i + 1
        })
    
    # 벡터 저장소에 추가
    store.add(
        ids=ids,
        embeddings=embeddings,
        documents=test_docs,
        metadatas=metadatas
    )
    
    print(f"✓ {len(test_docs)}개 문서 인덱싱 완료")
    print(f"  벡터 저장소 문서 수: {store.count()}")
    
    return store


def test_1_vector_retriever(store):
    """테스트 1: 벡터 검색"""
    print_separator("테스트 1: 벡터 검색")
    
    retriever = VectorRetriever(vector_store=store)
    
    query = "5G voice call quality"
    print(f"쿼리: {query}")
    
    results = retriever.search(query, k=3)
    print_results(results, "벡터 검색 결과")
    
    assert len(results) > 0, "검색 결과가 없습니다"
    assert all(hasattr(r, 'score') for r in results), "점수가 없습니다"
    print("\n✓ 벡터 검색 테스트 통과")


def test_2_bm25_retriever(store):
    """테스트 2: BM25 키워드 검색"""
    print_separator("테스트 2: BM25 키워드 검색")
    
    retriever = BM25Retriever(vector_store=store)
    
    query = "VoLTE interoperability"
    print(f"쿼리: {query}")
    
    results = retriever.search(query, k=3)
    print_results(results, "BM25 검색 결과")
    
    assert len(results) > 0, "검색 결과가 없습니다"
    
    # VoLTE 키워드가 포함되어 있는지 확인
    has_volte = any("VoLTE" in r.content for r in results)
    print(f"\n  VoLTE 키워드 포함 여부: {has_volte}")
    
    print("\n✓ BM25 검색 테스트 통과")


def test_3_hybrid_retriever_weighted(store):
    """테스트 3: 하이브리드 검색 (가중치 방식)"""
    print_separator("테스트 3: 하이브리드 검색 (Weighted)")
    
    retriever = HybridRetriever(
        vector_store=store,
        alpha=0.7,  # 벡터:BM25 = 7:3
        fusion_method="weighted"
    )
    
    query = "5G VoLTE quality testing"
    print(f"쿼리: {query}")
    print(f"가중치: 벡터={retriever.alpha}, BM25={1-retriever.alpha}")
    
    results = retriever.search(query, k=5)
    print_results(results, "하이브리드 검색 결과 (Weighted)")
    
    assert len(results) > 0, "검색 결과가 없습니다"
    print("\n✓ 하이브리드 검색 (Weighted) 테스트 통과")


def test_4_hybrid_retriever_rrf(store):
    """테스트 4: 하이브리드 검색 (RRF 방식)"""
    print_separator("테스트 4: 하이브리드 검색 (RRF)")
    
    retriever = HybridRetriever(
        vector_store=store,
        alpha=0.7,
        fusion_method="rrf"
    )
    
    query = "5G network architecture"
    print(f"쿼리: {query}")
    print(f"결합 방식: RRF (Reciprocal Rank Fusion)")
    
    results = retriever.search(query, k=5)
    print_results(results, "하이브리드 검색 결과 (RRF)")
    
    assert len(results) > 0, "검색 결과가 없습니다"
    print("\n✓ 하이브리드 검색 (RRF) 테스트 통과")


def test_5_reranker(store):
    """테스트 5: 리랭킹"""
    print_separator("테스트 5: 리랭킹")
    
    # 먼저 벡터 검색으로 후보 추출
    retriever = VectorRetriever(vector_store=store)
    query = "femtocell small cell deployment"
    
    print(f"쿼리: {query}")
    
    # 초기 검색 (더 많은 후보)
    initial_results = retriever.search(query, k=8)
    print_results(initial_results[:5], "초기 검색 결과 (상위 5개)")
    
    # 리랭킹
    reranker = Reranker(method="bm25")
    reranked_results = reranker.rerank(query, initial_results, top_k=5)
    print_results(reranked_results, "리랭킹 후 결과")
    
    assert len(reranked_results) > 0, "리랭킹 결과가 없습니다"
    
    # 순서가 바뀌었는지 확인
    initial_ids = [r.id for r in initial_results[:5]]
    reranked_ids = [r.id for r in reranked_results]
    
    if initial_ids != reranked_ids:
        print("\n  ✓ 리랭킹으로 순서가 변경됨")
    else:
        print("\n  - 순서 변경 없음 (검색 결과가 이미 최적)")
    
    print("\n✓ 리랭킹 테스트 통과")


def test_6_advanced_retriever(store):
    """테스트 6: 고급 검색기 (하이브리드 + 리랭킹)"""
    print_separator("테스트 6: 고급 검색기 (Advanced)")
    
    retriever = AdvancedRetriever(
        vector_store=store,
        alpha=0.6,
        fusion_method="rrf",
        rerank_method="bm25",
        use_rerank=True
    )
    
    query = "5G interoperability test femtocell"
    print(f"쿼리: {query}")
    print(f"설정: 하이브리드(RRF) + BM25 리랭킹")
    
    results = retriever.search(query, k=5)
    print_results(results, "고급 검색 결과")
    
    assert len(results) > 0, "검색 결과가 없습니다"
    print("\n✓ 고급 검색기 테스트 통과")


def test_7_comparison(store):
    """테스트 7: 검색 방법 비교"""
    print_separator("테스트 7: 검색 방법 비교")
    
    query = "VoLTE 5G voice quality"
    print(f"쿼리: {query}\n")
    
    # 1. 벡터만
    vector_retriever = VectorRetriever(vector_store=store)
    vector_results = vector_retriever.search(query, k=3)
    print("\n[벡터 검색 결과]")
    for i, r in enumerate(vector_results, 1):
        print(f"{i}. (점수: {r.score:.4f}) {r.content[:80]}...")
    
    # 2. BM25만
    bm25_retriever = BM25Retriever(vector_store=store)
    bm25_results = bm25_retriever.search(query, k=3)
    print("\n[BM25 검색 결과]")
    for i, r in enumerate(bm25_results, 1):
        print(f"{i}. (점수: {r.score:.4f}) {r.content[:80]}...")
    
    # 3. 하이브리드
    hybrid_retriever = HybridRetriever(vector_store=store, fusion_method="rrf")
    hybrid_results = hybrid_retriever.search(query, k=3)
    print("\n[하이브리드 검색 결과 (RRF)]")
    for i, r in enumerate(hybrid_results, 1):
        print(f"{i}. (점수: {r.score:.4f}) {r.content[:80]}...")
    
    # 4. 고급 (하이브리드 + 리랭킹)
    advanced_retriever = AdvancedRetriever(vector_store=store, use_rerank=True)
    advanced_results = advanced_retriever.search(query, k=3)
    print("\n[고급 검색 결과 (하이브리드 + 리랭킹)]")
    for i, r in enumerate(advanced_results, 1):
        print(f"{i}. (점수: {r.score:.4f}) {r.content[:80]}...")
    
    print("\n✓ 비교 테스트 완료")


def test_8_factory_function():
    """테스트 8: 팩토리 함수"""
    print_separator("테스트 8: 팩토리 함수")
    
    # 다양한 타입으로 검색기 생성
    retrievers = {
        "vector": get_retriever("vector"),
        "bm25": get_retriever("bm25"),
        "hybrid": get_retriever("hybrid", alpha=0.7),
        "advanced": get_retriever("advanced", use_rerank=True)
    }
    
    for name, retriever in retrievers.items():
        print(f"✓ {name} retriever: {type(retriever).__name__}")
    
    print("\n✓ 팩토리 함수 테스트 통과")


def main():
    """모든 테스트 실행"""
    print_separator("Retriever 모듈 테스트 시작")
    
    try:
        # 테스트 데이터 준비
        store = setup_test_data()
        
        # 각 테스트 실행
        test_1_vector_retriever(store)
        test_2_bm25_retriever(store)
        test_3_hybrid_retriever_weighted(store)
        test_4_hybrid_retriever_rrf(store)
        test_5_reranker(store)
        test_6_advanced_retriever(store)
        test_7_comparison(store)
        test_8_factory_function()
        
        # 정리
        print_separator("모든 테스트 완료")
        print("✅ 모든 테스트가 성공적으로 통과했습니다!")
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
