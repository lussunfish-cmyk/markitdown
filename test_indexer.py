#!/usr/bin/env python3
"""
indexer.py 모듈의 기능을 테스트합니다.
"""

import sys
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.indexer import (
    DocumentIndexer,
    IndexStateManager,
    create_indexer,
    get_indexer
)
from app.config import config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_state_manager():
    """IndexStateManager 테스트."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: IndexStateManager")
    logger.info("="*70)
    
    state_manager = IndexStateManager()
    
    # 테스트 파일
    test_file = Path("./output/test.md")
    
    # 초기 상태 확인
    logger.info(f"인덱싱 여부: {state_manager.is_indexed(test_file)}")
    
    # 파일 추가
    state_manager.add_file(
        file_path=test_file,
        chunk_ids=["chunk_1", "chunk_2", "chunk_3"],
        num_chunks=3,
        status="indexed"
    )
    logger.info("✓ 파일 추가됨")
    
    # 상태 확인
    logger.info(f"인덱싱 여부: {state_manager.is_indexed(test_file)}")
    file_info = state_manager.get_file_info(test_file)
    logger.info(f"파일 정보: {file_info}")
    
    # 모든 파일 목록
    indexed_files = state_manager.get_indexed_files()
    logger.info(f"인덱싱된 파일 수: {len(indexed_files)}")
    
    logger.info("✓ IndexStateManager 테스트 완료\n")


def test_indexer_single_file():
    """단일 파일 인덱싱 테스트."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: 단일 파일 인덱싱")
    logger.info("="*70)
    
    indexer = create_indexer()
    
    # output 디렉토리의 첫 번째 마크다운 파일 찾기
    output_dir = config.INDEXING.DOCUMENT_DIR
    md_files = list(output_dir.glob("*.md"))
    
    if not md_files:
        logger.warning("테스트할 마크다운 파일이 없음")
        logger.info("샘플 파일 생성 중...")
        
        # 샘플 파일 생성
        sample_file = output_dir / "sample_test.md"
        sample_file.write_text("""# 테스트 문서

이것은 인덱서 테스트를 위한 샘플 문서입니다.

## 섹션 1

5G 네트워크는 차세대 이동통신 기술입니다.

## 섹션 2

높은 속도와 낮은 지연시간이 특징입니다.
""", encoding='utf-8')
        logger.info(f"✓ 샘플 파일 생성: {sample_file.name}")
        md_files = [sample_file]
    
    test_file = md_files[0]
    logger.info(f"테스트 파일: {test_file.name}")
    
    # 파일 인덱싱
    result = indexer.index_file(test_file)
    
    logger.info(f"인덱싱 결과:")
    logger.info(f"  상태: {result['status']}")
    logger.info(f"  청크 수: {result['chunks']}")
    logger.info(f"  메시지: {result['message']}")
    
    # 재인덱싱 시도 (스킵되어야 함)
    logger.info("\n재인덱싱 시도 (변경 없음)...")
    result2 = indexer.index_file(test_file)
    logger.info(f"  상태: {result2['status']}")
    logger.info(f"  메시지: {result2['message']}")
    
    # 강제 재인덱싱
    logger.info("\n강제 재인덱싱...")
    result3 = indexer.index_file(test_file, force_reindex=True)
    logger.info(f"  상태: {result3['status']}")
    logger.info(f"  청크 수: {result3['chunks']}")
    
    logger.info("✓ 단일 파일 인덱싱 테스트 완료\n")


def test_indexer_directory():
    """디렉토리 인덱싱 테스트."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: 디렉토리 배치 인덱싱")
    logger.info("="*70)
    
    indexer = create_indexer()
    
    # output 디렉토리 모든 파일 인덱싱
    result = indexer.index_directory()
    
    logger.info(f"\n인덱싱 결과:")
    logger.info(f"  총 파일: {result.total_files}")
    logger.info(f"  인덱싱됨: {result.indexed_files}")
    logger.info(f"  실패: {result.failed_files}")
    logger.info(f"  총 청크: {result.total_chunks}")
    
    logger.info(f"\n파일 상세:")
    for file_info in result.files[:5]:  # 최대 5개만 출력
        logger.info(f"  - {file_info['filename']}: {file_info['status']} ({file_info['chunks']} 청크)")
    
    if len(result.files) > 5:
        logger.info(f"  ... 외 {len(result.files) - 5}개")
    
    logger.info("✓ 디렉토리 인덱싱 테스트 완료\n")


def test_get_indexed_files():
    """인덱싱된 파일 목록 조회 테스트."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: 인덱싱된 파일 목록")
    logger.info("="*70)
    
    indexer = get_indexer()
    
    documents = indexer.get_indexed_files()
    
    logger.info(f"인덱싱된 문서 수: {len(documents)}")
    
    for i, doc in enumerate(documents[:5], 1):
        logger.info(f"{i}. {doc.filename}")
        logger.info(f"   ID: {doc.id}")
        logger.info(f"   청크: {doc.total_chunks}")
        logger.info(f"   상태: {doc.status}")
        logger.info(f"   인덱싱 시각: {doc.indexed_at}")
    
    if len(documents) > 5:
        logger.info(f"... 외 {len(documents) - 5}개")
    
    logger.info("✓ 파일 목록 조회 테스트 완료\n")


def test_indexer_stats():
    """인덱싱 통계 테스트."""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: 인덱싱 통계")
    logger.info("="*70)
    
    indexer = get_indexer()
    
    stats = indexer.get_stats()
    
    logger.info("인덱싱 통계:")
    logger.info(f"  총 파일: {stats['total_files']}")
    logger.info(f"  총 청크: {stats['total_chunks']}")
    logger.info(f"  벡터 스토어 항목: {stats['vector_store_count']}")
    logger.info(f"  상태별 집계: {stats['status']}")
    
    logger.info(f"\n파일 목록 ({len(stats['files'])}개):")
    for filename in stats['files'][:10]:
        logger.info(f"  - {filename}")
    
    if len(stats['files']) > 10:
        logger.info(f"  ... 외 {len(stats['files']) - 10}개")
    
    logger.info("✓ 통계 테스트 완료\n")


def test_delete_document():
    """문서 삭제 테스트."""
    logger.info("\n" + "="*70)
    logger.info("TEST 6: 문서 삭제")
    logger.info("="*70)
    
    indexer = get_indexer()
    
    # 샘플 파일이 있으면 삭제 테스트
    sample_file = config.INDEXING.DOCUMENT_DIR / "sample_test.md"
    
    if sample_file.exists():
        logger.info(f"삭제할 파일: {sample_file.name}")
        
        # 먼저 인덱싱되어 있는지 확인
        result = indexer.index_file(sample_file, force_reindex=True)
        logger.info(f"인덱싱 상태: {result['status']} ({result['chunks']} 청크)")
        
        # 삭제
        success = indexer.delete_document(sample_file)
        logger.info(f"삭제 결과: {'성공' if success else '실패'}")
        
        # 삭제 확인
        state_manager = indexer.state_manager
        is_indexed = state_manager.is_indexed(sample_file)
        logger.info(f"인덱싱 상태: {is_indexed}")
        
        logger.info("✓ 문서 삭제 테스트 완료\n")
    else:
        logger.warning("삭제 테스트용 샘플 파일이 없음")
        logger.info("✓ 문서 삭제 테스트 스킵\n")


def main():
    """모든 테스트 실행."""
    logger.info("\n" + "="*70)
    logger.info("INDEXER 테스트 시작")
    logger.info("="*70)
    
    try:
        # 1. 상태 관리자 테스트
        test_state_manager()
        
        # 2. 단일 파일 인덱싱
        test_indexer_single_file()
        
        # 3. 디렉토리 배치 인덱싱
        test_indexer_directory()
        
        # 4. 인덱싱된 파일 목록
        test_get_indexed_files()
        
        # 5. 통계
        test_indexer_stats()
        
        # 6. 문서 삭제
        test_delete_document()
        
        logger.info("\n" + "="*70)
        logger.info("✓ 모든 테스트 완료!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"\n✗ 테스트 실패: {str(e)}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
