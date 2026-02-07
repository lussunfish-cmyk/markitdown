#!/usr/bin/env python3
"""
RAG 파이프라인 테스트 스크립트.

Phase 2: 성능 개선 테스트 포함
- 캐싱 테스트
- 스트리밍 테스트
- 메트릭 측정 테스트
"""

import sys
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.rag import RAGPipeline, create_rag_pipeline, get_rag_pipeline
from app.vector_store import get_vector_store
from app.embedding import DocumentEmbedder

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_test_data():
    """테스트용 샘플 문서를 벡터 스토어에 추가합니다."""
    logger.info("\n" + "="*70)
    logger.info("테스트 데이터 설정")
    logger.info("="*70)
    
    # 샘플 문서
    sample_docs = [
        {
            "content": """5G 기술 개요
5G는 5세대 이동통신 기술로, 4G LTE보다 훨씬 빠른 속도와 낮은 지연시간을 제공합니다.
주요 특징:
- 최대 다운로드 속도: 20Gbps
- 지연시간: 1ms 이하
- 동시 연결: 1km² 당 100만 개 디바이스
5G는 IoT, 자율주행, 스마트시티 등 다양한 분야에 활용됩니다.""",
            "source": "5G_technology.md",
            "doc_id": "doc_5g"
        },
        {
            "content": """LTE 기술 설명
LTE(Long Term Evolution)는 4세대 이동통신 기술입니다.
주요 사양:
- 다운로드 속도: 최대 300Mbps
- 업로드 속도: 최대 75Mbps
- 지연시간: 10-20ms
LTE는 전 세계적으로 가장 널리 사용되는 모바일 네트워크 기술입니다.""",
            "source": "LTE_overview.md",
            "doc_id": "doc_lte"
        },
        {
            "content": """VoLTE (Voice over LTE)
VoLTE는 LTE 네트워크를 통해 음성 통화를 전송하는 기술입니다.
장점:
- 향상된 음질 (HD Voice)
- 빠른 호 연결 속도
- 데이터와 음성 동시 사용 가능
- 배터리 효율 개선
VoLTE는 기존 회선 교환 방식보다 효율적인 패킷 교환 방식을 사용합니다.""",
            "source": "VoLTE_guide.md",
            "doc_id": "doc_volte"
        },
        {
            "content": """Python 프로그래밍 기초
Python은 간결하고 읽기 쉬운 문법을 가진 고급 프로그래밍 언어입니다.
주요 특징:
- 동적 타이핑
- 가비지 컬렉션
- 풍부한 표준 라이브러리
- 멀티 패러다임 (객체지향, 함수형, 절차형)
Python은 웹 개발, 데이터 과학, 인공지능, 자동화 등에 널리 사용됩니다.""",
            "source": "Python_basics.md",
            "doc_id": "doc_python"
        }
    ]
    
    try:
        # 벡터 스토어 초기화
        vector_store = get_vector_store()
        
        # 기존 데이터 삭제 (테스트용)
        logger.info("기존 테스트 데이터 삭제...")
        for doc in sample_docs:
            try:
                vector_store.delete_by_source(doc["source"])
            except:
                pass
        
        # 임베더 초기화
        embedder = DocumentEmbedder()
        
        # 각 문서 추가
        for doc in sample_docs:
            logger.info(f"문서 추가: {doc['source']}")
            
            # 청크 생성 및 임베딩 (한번에 처리)
            chunks = embedder.embed_document(
                content=doc["content"],
                source=doc["source"],
                show_progress=False
            )
            
            if not chunks:
                logger.warning(f"  ⚠️  청크 생성 실패: {doc['source']}")
                continue
            
            # 벡터 스토어에 추가
            ids = [chunk.id for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata.model_dump() for chunk in chunks]
            
            vector_store.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"  ✓ {len(chunks)}개 청크 추가됨")
        
        # 확인
        total_count = vector_store.count()
        logger.info(f"\n총 {total_count}개 청크가 벡터 스토어에 저장됨")
        
        return True
    
    except Exception as e:
        logger.error(f"테스트 데이터 설정 실패: {str(e)}")
        return False


def test_1_basic_query():
    """기본 질의응답 테스트"""
    logger.info("\n" + "="*70)
    logger.info("테스트 1: 기본 질의응답")
    logger.info("="*70)
    
    try:
        # RAG 파이프라인 생성
        rag = create_rag_pipeline(retriever_type="vector")
        
        # 질문
        question = "5G의 최대 다운로드 속도는?"
        logger.info(f"\n질문: {question}")
        
        # 답변 생성
        result = rag.query(question, top_k=3)
        
        # 결과 출력
        logger.info(f"\n답변:\n{result.answer}")
        logger.info(f"\n사용된 청크 수: {result.num_chunks}")
        logger.info(f"\n출처 정보:")
        for source in result.sources:
            logger.info(f"  - {source}")
        
        assert result.answer, "답변이 생성되지 않았습니다"
        assert result.num_chunks > 0, "컨텍스트가 사용되지 않았습니다"
        
        logger.info("\n✅ 테스트 1 통과")
        return True
    
    except Exception as e:
        logger.error(f"\n❌ 테스트 1 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_2_advanced_retriever():
    """고급 검색기 사용 테스트"""
    logger.info("\n" + "="*70)
    logger.info("테스트 2: 고급 검색기 (하이브리드 + 리랭킹)")
    logger.info("="*70)
    
    try:
        # 고급 검색기로 RAG 파이프라인 생성
        rag = create_rag_pipeline(retriever_type="advanced")
        
        # 질문
        question = "VoLTE의 장점은 무엇인가요?"
        logger.info(f"\n질문: {question}")
        
        # 답변 생성
        result = rag.query(question, top_k=3)
        
        # 결과 출력
        logger.info(f"\n답변:\n{result.answer}")
        logger.info(f"\n사용된 청크 수: {result.num_chunks}")
        
        assert result.answer, "답변이 생성되지 않았습니다"
        
        logger.info("\n✅ 테스트 2 통과")
        return True
    
    except Exception as e:
        logger.error(f"\n❌ 테스트 2 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_3_no_results():
    """관련 문서가 없을 때 처리 테스트"""
    logger.info("\n" + "="*70)
    logger.info("테스트 3: 관련 문서 없음 처리")
    logger.info("="*70)
    
    try:
        rag = create_rag_pipeline(retriever_type="vector")
        
        # 전혀 관련 없는 질문
        question = "양자 컴퓨팅의 큐비트란 무엇인가요?"
        logger.info(f"\n질문: {question}")
        
        # 답변 생성
        result = rag.query(question, top_k=3)
        
        # 결과 출력
        logger.info(f"\n답변:\n{result.answer}")
        logger.info(f"\n사용된 청크 수: {result.num_chunks}")
        
        # 관련 문서가 없어도 적절한 응답이 반환되어야 함
        assert result.answer, "답변이 생성되지 않았습니다"
        
        logger.info("\n✅ 테스트 3 통과")
        return True
    
    except Exception as e:
        logger.error(f"\n❌ 테스트 3 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_4_custom_parameters():
    """커스텀 파라미터 테스트"""
    logger.info("\n" + "="*70)
    logger.info("테스트 4: 커스텀 파라미터")
    logger.info("="*70)
    
    try:
        # 커스텀 프롬프트로 RAG 생성
        custom_system_prompt = """당신은 친절한 기술 설명가입니다.
초보자도 이해할 수 있도록 쉽게 설명해주세요."""
        
        rag = create_rag_pipeline(
            retriever_type="vector",
            system_prompt=custom_system_prompt,
            top_k=2,
            temperature=0.3
        )
        
        # 질문
        question = "LTE란 무엇인가요?"
        logger.info(f"\n질문: {question}")
        
        # 답변 생성
        result = rag.query(question)
        
        # 결과 출력
        logger.info(f"\n답변:\n{result.answer}")
        logger.info(f"\nMetadata: {result.metadata}")
        
        assert result.answer, "답변이 생성되지 않았습니다"
        
        logger.info("\n✅ 테스트 4 통과")
        return True
    
    except Exception as e:
        logger.error(f"\n❌ 테스트 4 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_5_singleton_pattern():
    """싱글톤 패턴 테스트"""
    logger.info("\n" + "="*70)
    logger.info("테스트 5: 싱글톤 패턴")
    logger.info("="*70)
    
    try:
        # 두 번 호출해도 같은 인스턴스 반환
        rag1 = get_rag_pipeline()
        rag2 = get_rag_pipeline()
        
        assert rag1 is rag2, "싱글톤 패턴이 작동하지 않습니다"
        logger.info("✓ 동일한 인스턴스 반환 확인")
        
        # 새 인스턴스 생성
        rag3 = create_rag_pipeline()
        assert rag1 is not rag3, "create는 새 인스턴스를 생성해야 합니다"
        logger.info("✓ create_rag_pipeline은 새 인스턴스 생성 확인")
        
        logger.info("\n✅ 테스트 5 통과")
        return True
    
    except Exception as e:
        logger.error(f"\n❌ 테스트 5 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_6_chat_with_history():
    """대화 히스토리 테스트"""
    logger.info("\n" + "="*70)
    logger.info("테스트 6: 대화 히스토리")
    logger.info("="*70)
    
    try:
        rag = create_rag_pipeline(retriever_type="vector")
        
        # 첫 번째 질문
        question1 = "5G란 무엇인가요?"
        logger.info(f"\n질문 1: {question1}")
        result1 = rag.query(question1)
        logger.info(f"답변 1:\n{result1.answer[:200]}...")
        
        # 대화 히스토리 구성
        history = [
            {"role": "user", "content": question1},
            {"role": "assistant", "content": result1.answer}
        ]
        
        # 후속 질문
        question2 = "그것의 주요 특징은 무엇인가요?"
        logger.info(f"\n질문 2: {question2}")
        result2 = rag.chat(question2, conversation_history=history)
        logger.info(f"답변 2:\n{result2.answer[:200]}...")
        
        assert result1.answer, "첫 번째 답변이 생성되지 않았습니다"
        assert result2.answer, "두 번째 답변이 생성되지 않았습니다"
        
        logger.info("\n✅ 테스트 6 통과")
        return True
    
    except Exception as e:
        logger.error(f"\n❌ 테스트 6 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_7_metrics():
    """성능 메트릭 테스트 (Phase 2)"""
    logger.info("\n" + "="*70)
    logger.info("테스트 7: 성능 메트릭 (Phase 2)")
    logger.info("="*70)
    
    try:
        rag = create_rag_pipeline(retriever_type="vector")
        
        # 메트릭 포함하여 쿼리
        question = "LTE의 속도는?"
        logger.info(f"\n질문: {question}")
        
        result = rag.query(question, top_k=3, include_metrics=True)
        
        # 메트릭 확인
        assert result.metrics is not None, "메트릭이 생성되지 않았습니다"
        
        logger.info(f"\n메트릭:")
        logger.info(f"  전체 시간: {result.metrics.query_time:.3f}초")
        logger.info(f"  검색 시간: {result.metrics.search_time:.3f}초")
        logger.info(f"  LLM 시간: {result.metrics.llm_time:.3f}초")
        logger.info(f"  청크 수: {result.metrics.num_chunks}")
        logger.info(f"  컨텍스트 길이: {result.metrics.context_length}자")
        
        # 메트릭 검증
        assert result.metrics.query_time > 0, "쿼리 시간이 측정되지 않았습니다"
        assert result.metrics.search_time > 0, "검색 시간이 측정되지 않았습니다"
        assert result.metrics.llm_time > 0, "LLM 시간이 측정되지 않았습니다"
        assert result.metrics.num_chunks > 0, "청크가 사용되지 않았습니다"
        
        logger.info("\n✅ 테스트 7 통과")
        return True
    
    except Exception as e:
        logger.error(f"\n❌ 테스트 7 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_8_caching():
    """캐싱 테스트 (Phase 2)"""
    logger.info("\n" + "="*70)
    logger.info("테스트 8: 캐싱 (Phase 2)")
    logger.info("="*70)
    
    try:
        rag = create_rag_pipeline(retriever_type="vector", enable_cache=True)
        
        # 캐시 초기화
        RAGPipeline.clear_cache()
        
        # 첫 번째 쿼리 (캐시 미스)
        question = "5G의 지연시간은?"
        logger.info(f"\n첫 번째 쿼리: {question}")
        result1 = rag.query(question, include_metrics=True)
        time1 = result1.metrics.query_time if result1.metrics else 0
        
        # 두 번째 동일 쿼리 (캐시 히트)
        logger.info(f"\n두 번째 쿼리 (동일): {question}")
        result2 = rag.query(question, include_metrics=True)
        time2 = result2.metrics.query_time if result2.metrics else 0
        
        # 캐시 통계 확인
        stats = RAGPipeline.get_cache_stats()
        logger.info(f"\n캐시 통계:")
        logger.info(f"  캐시 히트: {stats['cache_hits']}")
        logger.info(f"  캐시 미스: {stats['cache_misses']}")
        logger.info(f"  히트율: {stats['hit_rate_percent']}%")
        logger.info(f"  캐시 크기: {stats['cache_size']}")
        
        logger.info(f"\n성능 비교:")
        logger.info(f"  첫 번째 쿼리: {time1:.3f}초")
        logger.info(f"  두 번째 쿼리: {time2:.3f}초")
        if time1 > 0 and time2 > 0:
            speedup = (time1 - time2) / time1 * 100
            logger.info(f"  성능 향상: {speedup:.1f}%")
        
        # 캐시가 작동했는지 확인
        assert stats['cache_hits'] >= 1, "캐시 히트가 발생하지 않았습니다"
        
        logger.info("\n✅ 테스트 8 통과")
        return True
    
    except Exception as e:
        logger.error(f"\n❌ 테스트 8 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_9_streaming():
    """스트리밍 응답 테스트 (Phase 2)"""
    logger.info("\n" + "="*70)
    logger.info("테스트 9: 스트리밍 응답 (Phase 2)")
    logger.info("="*70)
    
    try:
        rag = create_rag_pipeline(retriever_type="vector")
        
        question = "Python의 주요 특징은?"
        logger.info(f"\n질문: {question}")
        logger.info("\n스트리밍 답변:")
        
        # 스트리밍 쿼리
        full_answer = ""
        chunk_count = 0
        
        for chunk in rag.stream_query(question, top_k=3):
            if isinstance(chunk, str):
                full_answer += chunk
                chunk_count += 1
                # 처음 몇 청크만 출력
                if chunk_count <= 5:
                    logger.info(f"  청크 {chunk_count}: '{chunk}'")
        
        logger.info(f"\n총 {chunk_count}개 청크 수신")
        logger.info(f"전체 답변 길이: {len(full_answer)}자")
        logger.info(f"답변 미리보기:\n{full_answer[:200]}...")
        
        # 검증
        assert chunk_count > 0, "스트리밍 청크가 생성되지 않았습니다"
        assert len(full_answer) > 0, "답변이 비어있습니다"
        
        logger.info("\n✅ 테스트 9 통과")
        return True
    
    except Exception as e:
        logger.error(f"\n❌ 테스트 9 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("\n" + "="*70)
    logger.info("RAG 파이프라인 테스트 시작 (Phase 2 포함)")
    logger.info("="*70)
    
    # 테스트 데이터 설정
    if not setup_test_data():
        logger.error("테스트 데이터 설정 실패. 테스트 중단.")
        sys.exit(1)
    
    # 테스트 실행
    results = {
        "test_1_basic_query": test_1_basic_query(),
        "test_2_advanced_retriever": test_2_advanced_retriever(),
        "test_3_no_results": test_3_no_results(),
        "test_4_custom_parameters": test_4_custom_parameters(),
        "test_5_singleton_pattern": test_5_singleton_pattern(),
        "test_6_chat_with_history": test_6_chat_with_history(),
        "test_7_metrics": test_7_metrics(),
        "test_8_caching": test_8_caching(),
        "test_9_streaming": test_9_streaming(),
    }
    
    # 결과 요약
    logger.info("\n" + "="*70)
    logger.info("테스트 결과 요약")
    logger.info("="*70)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    # Phase 구분
    phase1_tests = ["test_1_basic_query", "test_2_advanced_retriever", "test_3_no_results",
                    "test_4_custom_parameters", "test_5_singleton_pattern", "test_6_chat_with_history"]
    phase2_tests = ["test_7_metrics", "test_8_caching", "test_9_streaming"]
    
    logger.info("\n[Phase 1 테스트]")
    for test_name in phase1_tests:
        result = results[test_name]
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"{test_name}: {status}")
    
    logger.info("\n[Phase 2 테스트 - 성능 개선]")
    for test_name in phase2_tests:
        result = results[test_name]
        status = "✅ 통과" if result else "❌ 실패"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n총 {total}개 테스트 중 {passed}개 통과")
    
    if passed == total:
        logger.info("\n🎉 모든 테스트 통과!")
        sys.exit(0)
    else:
        logger.error(f"\n⚠️  {total - passed}개 테스트 실패")
        sys.exit(1)
