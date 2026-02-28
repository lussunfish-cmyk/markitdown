#!/bin/bash

set -e

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MarkItDown 통합 테스트 시작 ===${NC}"

# 1. 벡터 저장소 초기화
echo -e "\n${BLUE}[1/5] 벡터 저장소 초기화${NC}"
rm -rf ./vector_store ./batch_state
mkdir -p ./vector_store ./batch_state ./input ./output
echo -e "${GREEN}✅ 벡터 저장소 초기화 완료${NC}"
sleep 1

echo -e "\n${BLUE}서버 상태 확인 (http://localhost:8000/health)${NC}"
if ! curl -s http://localhost:8000/health > /dev/null; then
  echo -e "${RED}❌ API 서버가 실행 중이 아닙니다. 먼저 아래 명령으로 서버를 실행하세요.${NC}"
  echo -e "${RED}python -m uvicorn app.converter:app --host 0.0.0.0 --port 8000 --loop asyncio${NC}"
  exit 1
fi
echo -e "${GREEN}✅ API 서버 실행 확인 완료${NC}"

# 2. 테스트 문서 생성
echo -e "\n${BLUE}[2/5] 테스트 문서 준비${NC}"
cp sample.doc ./input/
echo -e "${GREEN}✅ 테스트 문서 준비 완료${NC}"

# 3. 문서 변환 및 인덱싱
echo -e "\n${BLUE}[3/5] 문서 변환 및 인덱싱${NC}"
curl -X 'POST' \
  'http://0.0.0.0:8000/convert-folder' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'auto_index=true'
sleep 3
echo -e "${GREEN}✅ 문서 변환 및 인덱싱 완료${NC}"

# 4. 인덱싱 상태 확인
echo -e "\n${BLUE}[4/5] 인덱싱 상태 확인${NC}"
curl -s "http://localhost:8000/documents"
echo -e "${NC}${GREEN}✅ 문서 인덱싱 완료 ${NC}"
sleep 1

# 5. 검색 및 RAG 테스트
echo -e "\n${BLUE}[5/5] 검색 및 RAG 테스트${NC}"

# 검색에 사용되는 쿼리와 RAG 질의응답에 사용되는 쿼리는 동일하게 설정하여, 검색 결과가 RAG 응답에 어떻게 활용되는지 확인할 수 있도록 합니다.

# 검색 테스트
echo "검색 결과:"
SEARCH_RESULT=$(curl -X 'GET' \
  'http://0.0.0.0:8000/search?query=Full%20rate%20speech%20transcoding%20%EC%9D%B4%EB%9E%80%3F&top_k=3' \
  -H 'accept: application/json')
echo -e "${GREEN}✅ 검색 결과: $SEARCH_RESULT${NC}${NC}"
sleep 1

# RAG 테스트
echo "RAG 질의응답:"
curl -X 'POST' \
  'http://0.0.0.0:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "Full rate speech transcoding 이란?",
  "top_k": 3,
  "temperature": 0,
  "max_tokens": 0,
  "include_sources": true
}'

echo -e "\n${GREEN}=== 통합 테스트 완료 ===${NC}"