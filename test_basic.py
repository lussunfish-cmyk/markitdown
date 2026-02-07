#!/usr/bin/env python
"""
기본 모듈 테스트 스크립트
- config 설정 확인
- ollama_client 연결 테스트
- API 엔드포인트 테스트
"""

import sys
import json
import requests
from pathlib import Path

# 색상 정의
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_info(text):
    print(f"{YELLOW}ℹ️  {text}{RESET}")

def test_config_import():
    """config 모듈 import 테스트"""
    print_header("Step 1: Config 모듈 임포트")
    try:
        from app.config import config
        print_success("config.py 임포트 성공")
        
        print_info(f"Ollama Base URL: {config.OLLAMA.BASE_URL}")
        print_info(f"Embedding Model: {config.OLLAMA.EMBEDDING_MODEL}")
        print_info(f"LLM Model: {config.OLLAMA.LLM_MODEL}")
        print_info(f"Vector Store Type: {config.VECTOR_STORE.STORE_TYPE}")
        print_info(f"Vector Embedding Dim: {config.VECTOR_STORE.EMBEDDING_DIM}")
        print_info(f"Chunk Size: {config.CHUNKING.CHUNK_SIZE}")
        print_info(f"RAG Top K: {config.RAG.TOP_K}")
        
        return True
    except Exception as e:
        print_error(f"config 임포트 실패: {e}")
        return False

def test_schemas_import():
    """schemas 모듈 import 테스트"""
    print_header("Step 2: Schemas 모듈 임포트")
    try:
        from app.schemas import (
            EmbeddingRequest, EmbeddingResponse,
            RAGRequest, RAGResponse,
            IndexRequest, IndexResponse
        )
        print_success("schemas.py 임포트 성공")
        print_info("모든 주요 스키마 클래스 로드 완료")
        return True
    except Exception as e:
        print_error(f"schemas 임포트 실패: {e}")
        return False

def test_ollama_client():
    """OllamaClient 연결 테스트"""
    print_header("Step 3: Ollama 클라이언트 연결")
    try:
        from app.ollama_client import OllamaClient
        
        print_info("OllamaClient 초기화 중...")
        client = OllamaClient()
        print_success("Ollama 서버 연결 성공")
        
        # 모델 확인
        models = client.list_models()
        print_info(f"사용 가능한 모델: {models}")
        
        # 필수 모델 확인
        embedding_available = client.check_model_available(client.embedding_model)
        llm_available = client.check_model_available(client.llm_model)
        
        if embedding_available:
            print_success(f"Embedding 모델 '{client.embedding_model}' 사용 가능")
        else:
            print_error(f"Embedding 모델 '{client.embedding_model}' 미설치")
            
        if llm_available:
            print_success(f"LLM 모델 '{client.llm_model}' 사용 가능")
        else:
            print_error(f"LLM 모델 '{client.llm_model}' 미설치")
        
        return embedding_available and llm_available
    except Exception as e:
        print_error(f"Ollama 클라이언트 테스트 실패: {e}")
        return False

def test_api_health():
    """API 헬스 체크"""
    print_header("Step 4: API 헬스 체크")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        response.raise_for_status()
        data = response.json()
        print_success("API 헬스 체크 응답 수신")
        print_info(f"응답: {json.dumps(data, indent=2)}")
        return True
    except Exception as e:
        print_error(f"API 헬스 체크 실패: {e}")
        return False

def test_supported_formats():
    """지원 파일 형식 조회"""
    print_header("Step 5: 지원 파일 형식 조회")
    try:
        response = requests.get("http://localhost:8000/supported-formats", timeout=5)
        response.raise_for_status()
        data = response.json()
        print_success("지원 파일 형식 조회 성공")
        print_info(f"지원 형식 수: {data['count']}개")
        print_info(f"지원 형식: {', '.join(data['formats'][:10])}...")
        return True
    except Exception as e:
        print_error(f"지원 파일 형식 조회 실패: {e}")
        return False

def test_embedding():
    """임베딩 테스트"""
    print_header("Step 6: 임베딩 생성 테스트")
    try:
        from app.ollama_client import OllamaClient
        
        client = OllamaClient()
        test_text = "안녕하세요. 이것은 테스트입니다."
        
        print_info(f"테스트 텍스트: '{test_text}'")
        print_info("임베딩 생성 중...")
        
        embedding = client.embed(test_text)
        
        print_success("임베딩 생성 성공")
        print_info(f"임베딩 차원: {len(embedding)}")
        print_info(f"임베딩 샘플 (처음 5개): {embedding[:5]}")
        
        return len(embedding) > 0
    except Exception as e:
        print_error(f"임베딩 테스트 실패: {e}")
        return False

def test_ollama_generate():
    """Ollama 텍스트 생성 테스트"""
    print_header("Step 7: LLM 텍스트 생성 테스트")
    try:
        from app.ollama_client import OllamaClient
        
        client = OllamaClient()
        prompt = "한국의 수도는?"
        
        print_info(f"프롬프트: '{prompt}'")
        print_info("응답 생성 중...")
        
        response = client.generate(prompt, temperature=0.3, num_predict=50)
        
        print_success("텍스트 생성 성공")
        print_info(f"응답: {response}")
        
        return len(response) > 0
    except Exception as e:
        print_error(f"텍스트 생성 테스트 실패: {e}")
        return False

def main():
    print(f"\n{BLUE}{'='*60}")
    print(f"  MarkItDown RAG 기본 모듈 테스트")
    print(f"{'='*60}{RESET}\n")
    
    results = []
    
    # 테스트 실행
    results.append(("Config Import", test_config_import()))
    results.append(("Schemas Import", test_schemas_import()))
    results.append(("Ollama Client", test_ollama_client()))
    results.append(("API Health Check", test_api_health()))
    results.append(("Supported Formats", test_supported_formats()))
    results.append(("Embedding Test", test_embedding()))
    results.append(("LLM Generate Test", test_ollama_generate()))
    
    # 결과 요약
    print_header("테스트 결과 요약")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = f"{GREEN}✅ PASS{RESET}" if result else f"{RED}❌ FAIL{RESET}"
        print(f"{test_name:<30} {status}")
    
    print(f"\n총 테스트: {total}개")
    print(f"{GREEN}성공: {passed}개{RESET}")
    print(f"{RED}실패: {total - passed}개{RESET}")
    
    if passed == total:
        print(f"\n{GREEN}🎉 모든 테스트 통과!{RESET}")
        return 0
    else:
        print(f"\n{RED}⚠️  일부 테스트 실패{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
