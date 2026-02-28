import os
import sys
import logging
import difflib
import string
import pandas as pd
from typing import Optional, List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragas.testset import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from langchain_core.documents import Document as LangchainDocument

# 기존 scripts/generate_testset.py의 로직을 함수화하여 재사용 가능하게 변경
from .config import config

logger = logging.getLogger(__name__)
from .llm_client import get_llm_client

from .embedding import chunk_text
import ast
import re
from .retriever import get_retriever, SearchResult

async def generate_testset_logic(input_dir: str, output_file: str, test_size: Optional[int] = None) -> List[dict]:
    # 디렉토리 생성
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    logger.info(f"Loading documents from {input_dir}...")
    
    loader = DirectoryLoader(
        input_dir, 
        glob="**/*.md", 
        loader_cls=TextLoader,
        show_progress=False
    )
    documents = loader.load()
    
    if not documents:
        raise ValueError(f"No documents found in {input_dir}")

    # 문서가 너무 크면 임베딩 시 400 에러(context length exceeded)가 발생할 수 있으므로 미리 분할
    # RecursiveCharacterTextSplitter 대신 앱의 청킹 로직(MarkdownChunker) 사용 (인덱싱과 일관성 유지)
    chunked_docs = []
    for doc in documents:
        chunks = chunk_text(
            doc.page_content, 
            chunk_size=config.CHUNKING.CHUNK_SIZE,
            chunk_overlap=config.CHUNKING.CHUNK_OVERLAP,
            use_markdown=True
        )
        for chunk in chunks:
            chunked_docs.append(LangchainDocument(
                page_content=chunk,
                metadata=doc.metadata
            ))
    documents = chunked_docs

    # test_size 자동 계산
    if test_size is None:
        total_chars = sum(len(doc.page_content) for doc in documents)
        calculated_size = int(total_chars / 2000)
        test_size = max(5, min(calculated_size, 50))
        logger.info(f"Auto-calculated test_size: {test_size}")

    logger.info("Setting up LLM and Embeddings...")
    
    # Config에서 설정 가져오기
    ollama_base_url = config.OLLAMA.BASE_URL
    llm_model = config.OLLAMA.LLM_MODEL
    embedding_model = config.OLLAMA.EMBEDDING_MODEL

    generator_llm = ChatOllama(base_url=ollama_base_url, model=llm_model)
    critic_llm = ChatOllama(base_url=ollama_base_url, model=llm_model)
    embeddings = OllamaEmbeddings(base_url=ollama_base_url, model=embedding_model)
    
    try:
        # ragas v0.2+ (TestsetGenerator.from_langchain)
        # chunk_size=512로 제한하여 로컬 임베딩 모델 컨텍스트 초과 방지
        generator = TestsetGenerator.from_langchain(
            generator_llm=generator_llm,
            critic_llm=critic_llm,
            embeddings=embeddings,
            chunk_size=config.CHUNKING.CHUNK_SIZE
        )
        
        distributions = {
            # 질문의 품질(정확성)을 높이기 위해 단순 질문(simple)의 비중을 대폭 늘립니다.
            # 로컬 LLM 사용 시 복잡한 추론(multi_context)은 할루시네이션을 유발할 수 있습니다.
            simple: 0.8,
            reasoning: 0.2,
            multi_context: 0.0
        }

        logger.info(f"Generating testset (size={test_size})...")
        testset = generator.generate_with_langchain_docs(
            documents, 
            test_size=test_size,
            distributions=distributions
        )
        
        df = testset.to_pandas()
        df.to_csv(output_file, index=False)
        logger.info(f"Testset saved to {output_file}")
        
        # 미리보기 데이터 반환
        if 'question' in df.columns and 'ground_truth' in df.columns:
            return df[['question', 'ground_truth']].head().to_dict('records')
        return df.head().to_dict('records')

    except Exception as e:
        logger.error(f"Failed to generate testset: {str(e)}")
        raise

def normalize_text(text: str) -> str:
    """텍스트 정규화: 소문자 변환, 특수문자/구두점 제거, 연속 공백 제거"""
    text = str(text).lower()
    # 모든 구두점 및 특수문자를 공백으로 치환하여 비교 정확도 향상
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

async def evaluate_retrieval_logic(testset_path: str, top_k: int = 5) -> dict:
    if not os.path.exists(testset_path):
        raise FileNotFoundError(f"Testset file verify not found: {testset_path}")

    logger.info(f"Loading testset from {testset_path}...")
    df = pd.read_csv(testset_path)

    if not df.empty and 'question' in df.columns:
        logger.info(f"Loaded testset preview (first question of {len(df)} rows): {df.iloc[0]['question']}")
    else:
        logger.warning(f"Loaded testset from {testset_path} is empty or missing 'question' column.")
    
    if 'contexts' in df.columns:
        df['contexts'] = df['contexts'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    retriever = get_retriever()
    llm_client = get_llm_client()
    results = []
    
    logger.info("Starting evaluation...")
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    total_mrr = 0
    total_hit_rate = 0
    
    for idx, row in df.iterrows():
        question = row['question']
        ground_truth_contexts = row['ground_truth_context'] if 'ground_truth_context' in row else row.get('contexts', [])
        
        # AdvancedRetriever의 메서드는 search입니다.
        # 1. 쿼리 확장 (Query Expansion)
        # 원본 질문과 재작성된 쿼리를 모두 사용
        search_queries = [question]
        rewritten_query = None
        if config.RAG.ENABLE_QUERY_REWRITING:
            try:
                prompt = config.PROMPT.QUERY_REWRITE_TEMPLATE.format(question=question)
                response = llm_client.generate(prompt=prompt, temperature=0.1, num_predict=50)
                rewritten = response.strip().strip('"')
                if rewritten:
                    search_queries.append(rewritten)
                    rewritten_query = rewritten
                    logger.info(f"    Rewritten: '{question[:30]}...' -> '{rewritten}'")
            except Exception:
                pass # 재작성 실패 시 원본 사용

        # 다중 쿼리 검색 및 병합
        merged_results = {}
        # 쿼리 확장 시 각 쿼리별로 더 많은 후보를 가져와서 병합 (Recall 향상)
        candidate_k = top_k * 3
        for q in search_queries:
            results = retriever.search(q, k=candidate_k)
            for result in results:
                if result.id not in merged_results or result.score > merged_results[result.id].score:
                    merged_results[result.id] = result
        
        # 점수순 정렬 후 상위 k개 선택
        retrieved_docs = sorted(merged_results.values(), key=lambda x: x.score, reverse=True)[:top_k]
        retrieved_contents = [doc.content for doc in retrieved_docs]
        # 파싱 오류 방지를 위한 예외 처리 혹은 타입 체크
        if isinstance(ground_truth_contexts, str):
             # 혹시 문자열로 남아있다면 다시 파싱 시도 (csv read 시 종종 발생)
             try:
                 ground_truth_contexts = ast.literal_eval(ground_truth_contexts)
             except:
                 ground_truth_contexts = [ground_truth_contexts]
        
        first_relevant_rank = 0
        max_similarity = 0.0
        
        found_gt_indices = set()  # 찾은 정답(GT)의 인덱스 (Recall용)
        relevant_retrieved_count = 0  # 정답인 검색 문서 개수 (Precision용)

        # 검색된 문서 순위별로 정답 여부 확인 (MRR 계산용)
        for rank, ret_ctx in enumerate(retrieved_contents, 1):
            is_relevant_doc = False
            norm_ret = normalize_text(ret_ctx)
            
            for gt_idx, gt_ctx in enumerate(ground_truth_contexts):
                # 정규화된 텍스트로 비교 (대소문자/공백 무시)
                norm_gt = normalize_text(gt_ctx)
                is_match = False
                
                # 1. 포함 관계 확인
                if norm_gt and norm_ret and (norm_gt in norm_ret or norm_ret in norm_gt):
                    is_match = True
                
                # 2. 유사도 확인 (Fuzzy Matching) - 60% 이상 유사하면 정답 처리
                else:
                    similarity = difflib.SequenceMatcher(None, norm_gt, norm_ret).ratio()
                    if similarity > max_similarity:
                        max_similarity = similarity
                    
                    if similarity >= 0.55:
                        is_match = True
                
                if is_match:
                    found_gt_indices.add(gt_idx)
                    is_relevant_doc = True
            
            if is_relevant_doc:
                relevant_retrieved_count += 1
                if first_relevant_rank == 0:
                    first_relevant_rank = rank
        
        # 지표 계산 수정
        relevant_gt_count = len(ground_truth_contexts)
        
        # Recall: 전체 정답 중 몇 개를 찾았는가 (최대 1.0)
        recall = len(found_gt_indices) / relevant_gt_count if relevant_gt_count > 0 else 0.0
        
        # Precision: 가져온 문서 중 몇 개가 정답인가
        precision = relevant_retrieved_count / top_k
        
        # Hit Rate: 하나라도 찾았으면 1
        hit_rate = 1.0 if relevant_retrieved_count > 0 else 0.0
        
        # 디버깅: 매칭 실패 시 로그 출력
        if hit_rate == 0 and ground_truth_contexts:
            logger.warning(f"[Miss] Q: {question[:30]}... (Max Sim: {max_similarity:.4f})")
            if rewritten_query:
                logger.warning(f"       Rewritten: {rewritten_query}")
            logger.warning(f"  - GT (first): {ground_truth_contexts[0][:50]}...")
            logger.warning(f"    -> Norm GT: {normalize_text(ground_truth_contexts[0])[:50]}...")
            logger.warning(f"  - Ret (top1): {retrieved_contents[0][:50] if retrieved_contents else 'None'}...")
            logger.warning(f"    -> Norm Ret: {normalize_text(retrieved_contents[0])[:50] if retrieved_contents else 'None'}...")
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mrr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0
        
        total_recall += recall
        total_precision += precision
        total_f1 += f1
        total_mrr += mrr
        total_hit_rate += hit_rate
        
        results.append({
            "question": question,
            "rewritten_query": rewritten_query,
            "hits": relevant_retrieved_count,
            "hit_rate": hit_rate,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "mrr": mrr,
            "max_similarity": max_similarity,
            "retrieved_top1": retrieved_contents[0] if retrieved_contents else ""
        })
        
    avg_recall = total_recall / len(df)
    avg_precision = total_precision / len(df)
    avg_f1 = total_f1 / len(df)
    avg_mrr = total_mrr / len(df)
    avg_hit_rate = total_hit_rate / len(df)
    
    # 상세 결과 저장
    result_df = pd.DataFrame(results)
    output_path = testset_path.replace(".csv", "_results.csv")
    result_df.to_csv(output_path, index=False)
    logger.info(f"Detailed results saved to {output_path}")
    
    return {
        "total_questions": len(df),
        "avg_recall": avg_recall,
        "avg_precision": avg_precision,
        "avg_f1": avg_f1,
        "avg_mrr": avg_mrr,
        "avg_hit_rate": avg_hit_rate,
        "detail_file": output_path
    }
