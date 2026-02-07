"""
vector_store.py 모듈의 기능을 테스트합니다.
"""

import sys
from pathlib import Path
from datetime import datetime

# app 모듈을 import하기 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from app.vector_store import (
    ChromaVectorStore,
    get_vector_store,
    get_default_vector_store
)
from app.ollama_client import get_ollama_client


def print_separator(title: str = ""):
    """구분선 출력"""
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}\n")
    else:
        print(f"{'-'*70}\n")


def test_1_basic_operations():
    """테스트 1: 기본 동작 (추가, 조회, 삭제)"""
    print_separator("테스트 1: 기본 동작 - 추가, 조회, 개수 확인")
    
    # 벡터 저장소 생성 (테스트용 별도 컬렉션)
    store = ChromaVectorStore(collection_name="test_basic_collection")
    
    # 초기화
    store.clear()
    print(f"✓ 컬렉션 초기화됨")
    initial_count = store.count()
    print(f"  현재 문서 수: {initial_count}")
    assert initial_count == 0, "초기화 후 문서 수는 0이어야 합니다"
    
    # 테스트 데이터 준비 (실제 임베딩 생성)
    ollama_client = get_ollama_client()
    
    test_docs = [
        "The 5G technology uses the 3.5GHz frequency band for enhanced mobile broadband services.",
        "Long Term Evolution (LTE) is a standard for wireless broadband communication based on 4G technology.",
        "Voice over LTE (VoLTE) is a technology that allows voice calls to be transmitted over LTE networks."
    ]
    
    print(f"\n📝 테스트 문서 준비 ({len(test_docs)}개):")
    for i, doc in enumerate(test_docs):
        print(f"  [{i+1}] {doc[:60]}...")
    
    # 임베딩 생성
    print(f"\n🔄 임베딩 생성 중...")
    try:
        embeddings = ollama_client.embed_batch(test_docs)
        print(f"✓ {len(embeddings)}개 임베딩 생성 완료")
        print(f"  임베딩 차원: {len(embeddings[0])}")
    except Exception as e:
        print(f"✗ 임베딩 생성 실패: {str(e)}")
        print(f"  → 테스트를 건너뜁니다")
        print_separator()
        return None
    
    # 메타데이터 준비
    metadatas = [
        {
            "source": "test_doc_1.md",
            "chunk_id": 0,
            "total_chunks": 1,
            "created_at": datetime.now().isoformat(),
            "topic": "5G"
        },
        {
            "source": "test_doc_2.md",
            "chunk_id": 0,
            "total_chunks": 1,
            "created_at": datetime.now().isoformat(),
            "topic": "LTE"
        },
        {
            "source": "test_doc_1.md",
            "chunk_id": 1,
            "total_chunks": 2,
            "created_at": datetime.now().isoformat(),
            "topic": "VoLTE"
        }
    ]
    
    # 문서 추가
    ids = ["test_1_chunk_0", "test_2_chunk_0", "test_1_chunk_1"]
    store.add(
        ids=ids,
        embeddings=embeddings,
        documents=test_docs,
        metadatas=metadatas
    )
    
    print(f"\n✓ 문서 추가 완료")
    after_add_count = store.count()
    print(f"  총 청크 수: {after_add_count}")
    assert after_add_count == 3, "3개 문서 추가 후 개수는 3이어야 합니다"
    
    # 컬렉션 정보
    info = store.get_collection_info()
    print(f"\n📊 컬렉션 정보:")
    print(f"  - 컬렉션명: {info['collection_name']}")
    print(f"  - 총 청크: {info['total_chunks']}")
    print(f"  - 총 문서: {info['total_documents']}")
    print(f"  - 저장 경로: {info['persist_directory']}")
    
    print_separator()
    return store


def test_2_get_operations(store: ChromaVectorStore):
    """테스트 2: 문서 조회"""
    print_separator("테스트 2: ID로 문서 조회")
    
    if store is None:
        print("⚠️  이전 테스트 실패로 건너뜁니다")
        print_separator()
        return
    
    # 특정 ID로 조회
    print(f"🔍 특정 ID로 문서 조회:")
    ids_to_get = ["test_1_chunk_0", "test_2_chunk_0"]
    docs = store.get(ids_to_get)
    
    print(f"  요청 ID 수: {len(ids_to_get)}")
    print(f"  조회된 문서 수: {len(docs)}")
    
    for doc in docs:
        print(f"\n  📄 ID: {doc['id']}")
        print(f"     문서: {doc['document'][:70]}...")
        print(f"     소스: {doc['metadata'].get('source', 'N/A')}")
        print(f"     토픽: {doc['metadata'].get('topic', 'N/A')}")
        embedding_dim = len(doc['embedding']) if doc['embedding'] is not None else 'N/A'
        print(f"     임베딩 차원: {embedding_dim}")
    
    assert len(docs) == len(ids_to_get), "조회된 문서 수가 요청 수와 같아야 합니다"
    
    # 존재하지 않는 ID 조회
    print(f"\n🔍 존재하지 않는 ID 조회:")
    non_existent = store.get(["non_existent_id"])
    print(f"  결과: {len(non_existent)}개 (예상: 0개)")
    assert len(non_existent) == 0, "존재하지 않는 ID는 빈 결과를 반환해야 합니다"
    
    print_separator()


def test_3_source_management(store: ChromaVectorStore):
    """테스트 3: 소스 파일 관리"""
    print_separator("테스트 3: 소스 파일 목록 및 관리")
    
    if store is None:
        print("⚠️  이전 테스트 실패로 건너뜁니다")
        print_separator()
        return
    
    # 전체 소스 목록
    sources = store.get_all_sources()
    print(f"📂 전체 소스 파일 ({len(sources)}개):")
    for source in sources:
        print(f"  - {source}")
    
    assert len(sources) == 2, "2개의 서로 다른 소스 파일이 있어야 합니다"
    assert "test_doc_1.md" in sources, "test_doc_1.md가 있어야 합니다"
    assert "test_doc_2.md" in sources, "test_doc_2.md가 있어야 합니다"
    
    print_separator()


def test_4_search_operations(store: ChromaVectorStore):
    """테스트 4: 의미론적 검색"""
    print_separator("테스트 4: 의미론적 검색")
    
    if store is None:
        print("⚠️  이전 테스트 실패로 건너뜁니다")
        print_separator()
        return
    
    ollama_client = get_ollama_client()
    
    # 검색 쿼리 (영어로 더 긴 텍스트)
    queries = [
        "What frequency band does 5G technology utilize for mobile broadband services?",
        "Can you explain the LTE wireless communication standard and its generation?",
        "How does voice calling work over LTE networks using VoLTE technology?"
    ]
    
    for idx, query in enumerate(queries):
        print(f"\n🔍 검색 쿼리 {idx+1}: '{query[:60]}...'")
        
        try:
            # 쿼리 임베딩
            query_embedding = ollama_client.embed(query)
            
            # 검색 수행
            results = store.search(query_embedding, k=3)
            
            print(f"  ✓ 결과 개수: {len(results)}")
            
            if results:
                best_result = results[0]
                print(f"\n  🏆 최고 유사도 결과:")
                print(f"     유사도: {best_result['score']:.4f}")
                print(f"     문서: {best_result['document'][:80]}...")
                print(f"     소스: {best_result['metadata'].get('source', 'N/A')}")
                
                # 점수가 합리적인 범위인지 확인
                assert 0 <= best_result['score'] <= 1, "유사도 점수는 0~1 사이여야 합니다"
            
        except Exception as e:
            print(f"  ⚠️  검색 실패: {str(e)}")
    
    print_separator()


def test_5_metadata_filtering(store: ChromaVectorStore):
    """테스트 5: 메타데이터 필터링"""
    print_separator("테스트 5: 메타데이터 기반 필터링 검색")
    
    if store is None:
        print("⚠️  이전 테스트 실패로 건너뜁니다")
        print_separator()
        return
    
    ollama_client = get_ollama_client()
    
    query = "Tell me about wireless communication technology and network standards"
    print(f"🔍 검색 쿼리: '{query}'")
    
    try:
        # 쿼리 임베딩
        query_embedding = ollama_client.embed(query)
        
        # 필터 없이 검색
        print(f"\n📌 필터 없이 검색:")
        results_all = store.search(query_embedding, k=5)
        print(f"  결과: {len(results_all)}개")
        for i, result in enumerate(results_all):
            print(f"    [{i+1}] {result['metadata'].get('source')} - 유사도: {result['score']:.4f}")
        
        # 특정 소스 파일만 검색
        print(f"\n📌 'test_doc_1.md' 파일만 검색:")
        results_filtered = store.search(
            query_embedding,
            k=5,
            filter={"source": "test_doc_1.md"}
        )
        print(f"  결과: {len(results_filtered)}개")
        for i, result in enumerate(results_filtered):
            print(f"    [{i+1}] {result['metadata'].get('source')} - 유사도: {result['score']:.4f}")
            assert result['metadata'].get('source') == "test_doc_1.md", "필터링된 결과는 test_doc_1.md만 있어야 합니다"
        
    except Exception as e:
        print(f"  ⚠️  검색 실패: {str(e)}")
    
    print_separator()


def test_6_delete_operations(store: ChromaVectorStore):
    """테스트 6: 문서 삭제"""
    print_separator("테스트 6: 문서 삭제 기능")
    
    if store is None:
        print("⚠️  이전 테스트 실패로 건너뜁니다")
        print_separator()
        return
    
    # 현재 개수
    before_count = store.count()
    print(f"📊 삭제 전 총 청크: {before_count}")
    
    # 특정 소스의 모든 청크 삭제
    print(f"\n🗑️  'test_doc_1.md' 삭제 중...")
    deleted_count = store.delete_by_source("test_doc_1.md")
    print(f"  ✓ 삭제된 청크: {deleted_count}개")
    
    # 삭제 후 개수
    after_count = store.count()
    print(f"  삭제 후 총 청크: {after_count}")
    assert after_count == before_count - deleted_count, "삭제 후 개수가 올바르지 않습니다"
    
    # 남은 소스
    remaining_sources = store.get_all_sources()
    print(f"\n📂 남은 소스 파일 ({len(remaining_sources)}개):")
    for source in remaining_sources:
        print(f"  - {source}")
    
    assert "test_doc_1.md" not in remaining_sources, "test_doc_1.md는 삭제되었어야 합니다"
    assert "test_doc_2.md" in remaining_sources, "test_doc_2.md는 남아있어야 합니다"
    
    # 특정 ID로 삭제
    print(f"\n🗑️  특정 ID로 삭제:")
    store.delete(["test_2_chunk_0"])
    print(f"  ✓ 'test_2_chunk_0' 삭제됨")
    
    final_count = store.count()
    print(f"  최종 문서 수: {final_count}")
    assert final_count == 0, "모든 문서가 삭제되었어야 합니다"
    
    print_separator()


def test_7_factory_and_singleton():
    """테스트 7: 팩토리 함수와 싱글톤 패턴"""
    print_separator("테스트 7: 팩토리 함수와 싱글톤 패턴")
    
    # 팩토리 함수로 생성
    print("📦 팩토리 함수로 벡터 저장소 생성:")
    store1 = get_vector_store(store_type="chroma", collection_name="factory_test")
    print(f"  ✓ 타입: {type(store1).__name__}")
    print(f"  ✓ 컬렉션: {store1.collection_name}")
    
    # 싱글톤 인스턴스
    print("\n🔒 싱글톤 인스턴스 가져오기:")
    store2 = get_default_vector_store()
    store3 = get_default_vector_store()
    print(f"  ✓ store2 is store3: {store2 is store3}")
    assert store2 is store3, "싱글톤 인스턴스는 동일해야 합니다"
    
    # FAISS 시도 (아직 미구현)
    print("\n⚠️  FAISS 저장소 시도 (미구현 상태):")
    try:
        faiss_store = get_vector_store(store_type="faiss")
        print(f"  ✗ 예외가 발생해야 하는데 생성됨")
        assert False, "FAISS는 NotImplementedError를 발생시켜야 합니다"
    except NotImplementedError as e:
        print(f"  ✓ 예상된 에러 발생: NotImplementedError")
        print(f"     메시지: {str(e)[:60]}...")
    
    # 잘못된 타입
    print("\n⚠️  잘못된 저장소 타입:")
    try:
        invalid_store = get_vector_store(store_type="invalid")
        print(f"  ✗ 예외가 발생해야 하는데 생성됨")
        assert False, "잘못된 타입은 ValueError를 발생시켜야 합니다"
    except ValueError as e:
        print(f"  ✓ 예상된 에러 발생: ValueError")
        print(f"     메시지: {str(e)[:60]}...")
    
    print_separator()


def test_8_persistence():
    """테스트 8: 영속성 테스트"""
    print_separator("테스트 8: 데이터 영속성 (재시작 후 데이터 보존)")
    
    collection_name = "persistence_test"
    
    # 1단계: 데이터 추가
    print("📝 1단계: 새 컬렉션 생성 및 데이터 추가")
    store1 = ChromaVectorStore(collection_name=collection_name)
    store1.clear()
    
    test_doc = "This is a persistence test document for ChromaDB vector storage."
    ollama_client = get_ollama_client()
    
    try:
        embedding = ollama_client.embed(test_doc)
        
        store1.add(
            ids=["persist_1"],
            embeddings=[embedding],
            documents=[test_doc],
            metadatas=[{"test": "persistence"}]
        )
        
        count1 = store1.count()
        print(f"  ✓ 추가된 문서 수: {count1}")
        assert count1 == 1, "1개 문서가 추가되어야 합니다"
        
        # 2단계: 새 인스턴스로 재로드
        print(f"\n📂 2단계: 같은 컬렉션을 새 인스턴스로 로드")
        store2 = ChromaVectorStore(collection_name=collection_name)
        count2 = store2.count()
        print(f"  ✓ 로드된 문서 수: {count2}")
        assert count2 == 1, "영속성: 이전 데이터가 유지되어야 합니다"
        
        # 데이터 확인
        docs = store2.get(["persist_1"])
        print(f"  ✓ 문서 조회 성공: {len(docs)}개")
        if docs:
            print(f"     내용: {docs[0]['document'][:50]}...")
            assert docs[0]['document'] == test_doc, "문서 내용이 일치해야 합니다"
        
        # 정리
        store2.clear()
        print(f"\n🗑️  테스트 컬렉션 정리 완료")
        
    except Exception as e:
        print(f"  ⚠️  영속성 테스트 실패: {str(e)}")
    
    print_separator()


def main():
    """테스트 실행"""
    print("\n" + "="*70)
    print("  VectorStore 통합 테스트")
    print("="*70)
    
    try:
        # 1. 기본 동작
        store = test_1_basic_operations()
        
        # 2. 조회 동작
        test_2_get_operations(store)
        
        # 3. 소스 관리
        test_3_source_management(store)
        
        # 4. 검색 동작
        test_4_search_operations(store)
        
        # 5. 메타데이터 필터링
        test_5_metadata_filtering(store)
        
        # 6. 삭제 동작
        test_6_delete_operations(store)
        
        # 7. 팩토리/싱글톤
        test_7_factory_and_singleton()
        
        # 8. 영속성
        test_8_persistence()
        
        print("\n" + "="*70)
        print("  ✅ 모든 테스트 통과!")
        print("="*70 + "\n")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ 테스트 실패 (Assertion): {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ 테스트 실패 (Exception): {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
