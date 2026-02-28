"""
embedding.py 모듈의 기능을 테스트합니다.
"""

import sys
import csv
from pathlib import Path
from datetime import datetime

# app 모듈을 import하기 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from app.embedding import (
    TextChunker,
    MarkdownChunker,
    DocumentEmbedder,
    create_embedder,
    chunk_text
)
from app.config import config


def save_chunks_to_csv(chunks: list, filename: str) -> Path:
    """
    청크 데이터를 CSV 파일로 저장합니다.
    
    Args:
        chunks: 청크 데이터 목록 (str 또는 DocumentChunk)
        filename: 저장할 CSV 파일명
        
    Returns:
        저장된 파일 경로
    """
    # debug 디렉토리 설정 및 생성
    debug_dir = Path("./debug")
    
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = debug_dir / filename
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        # DocumentChunk 객체인지 문자열인지 확인
        if chunks and hasattr(chunks[0], 'content'):
            # DocumentChunk 객체인 경우
            fieldnames = ['chunk_id', 'content_length', 'content', 'source', 'chunk_num', 'total_chunks', 'created_at', 'embedding_dim']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for chunk in chunks:
                writer.writerow({
                    'chunk_id': chunk.id,
                    'content_length': len(chunk.content),
                    'content': chunk.content[:500],  # 처음 500자만
                    'source': chunk.metadata.source,
                    'chunk_num': chunk.metadata.chunk_id,
                    'total_chunks': chunk.metadata.total_chunks,
                    'created_at': chunk.metadata.created_at,
                    'embedding_dim': len(chunk.embedding) if chunk.embedding else 0
                })
        else:
            # 문자열 청크인 경우
            fieldnames = ['chunk_id', 'content_length', 'content']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for idx, chunk in enumerate(chunks):
                writer.writerow({
                    'chunk_id': idx,
                    'content_length': len(chunk),
                    'content': chunk[:500]  # 처음 500자만
                })
    
    print(f"✓ CSV 파일 저장: {filepath}")
    return filepath


def test_text_chunker():
    """TextChunker의 기본 동작을 테스트합니다."""
    print("\n" + "="*80)
    print("TEST 1: TextChunker 기본 동작")
    print("="*80)
    
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    text = """This is a test document.
It has multiple paragraphs.

This is the second paragraph.
It also has multiple sentences.

And here is the third paragraph."""
    
    chunks = chunker.split_text(text)
    
    print(f"\n원본 텍스트 길이: {len(text)} 문자")
    print(f"생성된 청크 수: {len(chunks)}")
    print(f"청크 크기: {chunker.chunk_size}, 오버랩: {chunker.chunk_overlap}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n[청크 {i+1}] ({len(chunk)} 문자)")
        print(f"  {chunk[:100]}..." if len(chunk) > 100 else f"  {chunk}")
    
    # CSV 저장
    save_chunks_to_csv(chunks, "test_1_text_chunker.csv")
    
    assert len(chunks) > 0, "청크가 생성되지 않았습니다"
    print("\n✓ TextChunker 테스트 통과")


def test_markdown_chunker():
    """MarkdownChunker의 마크다운 구조 인식을 테스트합니다."""
    print("\n" + "="*80)
    print("TEST 2: MarkdownChunker 마크다운 구조 인식")
    print("="*80)
    
    chunker = MarkdownChunker(chunk_size=200, chunk_overlap=50)
    
    markdown_text = """# 제목 1

첫 번째 섹션의 내용입니다.

## 소제목 1-1

소제목 아래의 내용입니다.

# 제목 2

두 번째 섹션의 내용입니다.

## 소제목 2-1

더 많은 내용이 있습니다.

### 상세 제목

매우 상세한 내용입니다."""
    
    chunks = chunker.split_text(markdown_text)
    
    print(f"\n원본 마크다운 길이: {len(markdown_text)} 문자")
    print(f"생성된 청크 수: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        # 헤더 추출
        first_line = chunk.split('\n')[0]
        print(f"\n[청크 {i+1}] ({len(chunk)} 문자)")
        print(f"  시작: {first_line}")
        if len(chunk) > 150:
            print(f"  미리보기: {chunk[:150]}...")
        else:
            print(f"  전체: {chunk}")
    
    # CSV 저장
    save_chunks_to_csv(chunks, "test_2_markdown_chunker.csv")
    
    assert len(chunks) > 0, "청크가 생성되지 않았습니다"
    print("\n✓ MarkdownChunker 테스트 통과")


def test_document_embedder():
    """DocumentEmbedder의 임베딩 생성을 테스트합니다."""
    print("\n" + "="*80)
    print("TEST 3: DocumentEmbedder 임베딩 생성")
    print("="*80)
    
    embedder = create_embedder(use_markdown=True)
    
    # 간단한 테스트 문서
    test_content = """# 테스트 문서

이것은 임베딩 테스트를 위한 문서입니다.

## 섹션 1

첫 번째 섹션의 내용입니다. 이 내용은 여러 문장으로 구성되어 있습니다.

## 섹션 2

두 번째 섹션의 내용입니다. 이것도 여러 문장이 있습니다."""
    
    source = "test_document.md"
    
    try:
        print("\n임베딩 생성 중...")
        chunks = embedder.embed_document(
            content=test_content,
            source=source,
            show_progress=True
        )
        
        print(f"\n생성된 DocumentChunk 수: {len(chunks)}")
        
        if chunks:
            first_chunk = chunks[0]
            print(f"\n[첫 번째 청크 정보]")
            print(f"  ID: {first_chunk.id}")
            print(f"  내용 길이: {len(first_chunk.content)} 문자")
            print(f"  내용 미리보기: {first_chunk.content[:100]}...")
            print(f"  메타데이터:")
            print(f"    - source: {first_chunk.metadata.source}")
            print(f"    - chunk_id: {first_chunk.metadata.chunk_id}")
            print(f"    - total_chunks: {first_chunk.metadata.total_chunks}")
            print(f"    - created_at: {first_chunk.metadata.created_at}")
            
            if first_chunk.embedding:
                print(f"  임베딩:")
                print(f"    - 차원: {len(first_chunk.embedding)}")
                print(f"    - 첫 5개 값: {first_chunk.embedding[:5]}")
                assert len(first_chunk.embedding) > 0, "임베딩이 비어있습니다"
            else:
                print("  ⚠ 임베딩이 None입니다")
        
        # CSV 저장
        save_chunks_to_csv(chunks, "test_3_document_embedder.csv")
        
        assert len(chunks) > 0, "DocumentChunk가 생성되지 않았습니다"
        print("\n✓ DocumentEmbedder 테스트 통과")
        
    except Exception as e:
        print(f"\n✗ DocumentEmbedder 테스트 실패: {str(e)}")
        raise


def test_real_markdown_file():
    """실제 마크다운 파일로 임베딩을 테스트합니다."""
    print("\n" + "="*80)
    print("TEST 4: 실제 마크다운 파일 임베딩")
    print("="*80)
    
    # 테스트 파일 경로
    test_file = Path("./output/ToR574_INT_VoLTE_ViLTE_interoperability_test_description_4G_5G_ph1_rev_clean.md")
    
    if not test_file.exists():
        print(f"\n⚠ 테스트 파일이 존재하지 않습니다: {test_file}")
        print("  테스트를 건너뜁니다.")
        return
    
    print(f"\n파일: {test_file.name}")
    
    # 파일 크기 확인
    file_size = test_file.stat().st_size
    print(f"파일 크기: {file_size:,} 바이트")
    
    # 파일 내용 일부 읽기
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"텍스트 길이: {len(content):,} 문자")
    print(f"줄 수: {content.count(chr(10)):,}")
    
    # 임베딩 생성
    embedder = create_embedder(use_markdown=True)
    
    try:
        print("\n임베딩 생성 시작...")
        chunks = embedder.embed_document_from_file(
            file_path=test_file,
            show_progress=True
        )
        
        print(f"\n결과:")
        print(f"  총 청크 수: {len(chunks)}")
        
        if chunks:
            # 통계 계산
            chunk_lengths = [len(chunk.content) for chunk in chunks]
            avg_length = sum(chunk_lengths) / len(chunk_lengths)
            min_length = min(chunk_lengths)
            max_length = max(chunk_lengths)
            
            print(f"  청크 길이 통계:")
            print(f"    - 평균: {avg_length:.0f} 문자")
            print(f"    - 최소: {min_length} 문자")
            print(f"    - 최대: {max_length} 문자")
            
            # 임베딩 차원 확인
            if chunks[0].embedding:
                embedding_dim = len(chunks[0].embedding)
                print(f"  임베딩 차원: {embedding_dim}")
                
                expected_dim = config.VECTOR_STORE.EMBEDDING_DIM
                if embedding_dim == expected_dim:
                    print(f"  ✓ 예상 차원({expected_dim})과 일치")
                else:
                    print(f"  ⚠ 예상 차원({expected_dim})과 불일치")
            
            # 샘플 청크 출력
            print(f"\n[청크 샘플]")
            sample_indices = [0, len(chunks)//2, len(chunks)-1] if len(chunks) > 2 else range(len(chunks))
            
            for idx in sample_indices:
                chunk = chunks[idx]
                print(f"\n  청크 #{idx + 1}:")
                print(f"    ID: {chunk.id}")
                print(f"    길이: {len(chunk.content)} 문자")
                print(f"    내용: {chunk.content[:100]}...")
                if chunk.embedding:
                    print(f"    임베딩 차원: {len(chunk.embedding)}")
                    print(f"    임베딩 샘플: [{chunk.embedding[0]:.6f}, {chunk.embedding[1]:.6f}, ...]")
        
        # CSV 저장
        save_chunks_to_csv(chunks, "test_4_real_markdown_file.csv")
        
        assert len(chunks) > 0, "청크가 생성되지 않았습니다"
        print("\n✓ 실제 파일 임베딩 테스트 통과")
        
    except Exception as e:
        print(f"\n✗ 실제 파일 임베딩 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def test_chunk_text_helper():
    """chunk_text 헬퍼 함수를 테스트합니다."""
    print("\n" + "="*80)
    print("TEST 5: chunk_text 헬퍼 함수")
    print("="*80)
    
    text = "# Title\n\nSome content here.\n\n## Subtitle\n\nMore content."
    
    # 마크다운 청킹 (chunk_overlap을 명시적으로 설정)
    md_chunks = chunk_text(text, chunk_size=50, chunk_overlap=10, use_markdown=True)
    print(f"\n마크다운 청킹 결과: {len(md_chunks)} 청크")
    for i, chunk in enumerate(md_chunks):
        print(f"  청크 {i+1}: {chunk[:40]}...")
    
    # 일반 청킹 (chunk_overlap을 명시적으로 설정)
    normal_chunks = chunk_text(text, chunk_size=50, chunk_overlap=10, use_markdown=False)
    print(f"\n일반 청킹 결과: {len(normal_chunks)} 청크")
    for i, chunk in enumerate(normal_chunks):
        print(f"  청크 {i+1}: {chunk[:40]}...")
    
    # CSV 저장
    save_chunks_to_csv(md_chunks, "test_5_markdown_chunks.csv")
    save_chunks_to_csv(normal_chunks, "test_5_normal_chunks.csv")
    
    assert len(md_chunks) > 0, "마크다운 청크가 생성되지 않았습니다"
    assert len(normal_chunks) > 0, "일반 청크가 생성되지 않았습니다"
    print("\n✓ chunk_text 헬퍼 함수 테스트 통과")


def main():
    """모든 테스트를 실행합니다."""
    print("\n" + "="*80)
    print("EMBEDDING.PY 테스트 시작")
    print("="*80)
    
    tests = [
        ("TextChunker 기본 동작", test_text_chunker),
        ("MarkdownChunker 구조 인식", test_markdown_chunker),
        ("DocumentEmbedder 임베딩 생성", test_document_embedder),
        ("실제 파일 임베딩", test_real_markdown_file),
        ("chunk_text 헬퍼 함수", test_chunk_text_helper),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} 실패: {str(e)}")
            failed += 1
    
    print("\n" + "="*80)
    print("테스트 결과 요약")
    print("="*80)
    print(f"총 테스트: {len(tests)}")
    print(f"통과: {passed}")
    print(f"실패: {failed}")
    
    # CSV 파일 목록 확인
    debug_dir = Path("./debug")
    
    csv_files = list(debug_dir.glob("test_*.csv"))
    
    if csv_files:
        print(f"\n생성된 CSV 파일:")
        for csv_file in sorted(csv_files):
            file_size = csv_file.stat().st_size
            print(f"  - {csv_file.name} ({file_size:,} 바이트)")
    
    if failed == 0:
        print("\n✓ 모든 테스트 통과!")
    else:
        print(f"\n✗ {failed}개 테스트 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
