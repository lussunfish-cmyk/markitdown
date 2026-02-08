from typing import List, Optional
from pydantic import BaseModel, Field

class TestsetGenerateRequest(BaseModel):
    test_size: Optional[int] = Field(None, description="생성할 테스트셋 데이터(질문-답변 쌍)의 개수. 지정하지 않으면 문서 양에 따라 자동 계산됩니다.")
    input_dir: str = Field("/app/output", description="테스트셋 생성에 사용할 문서가 있는 디렉토리 경로")
    output_file: str = Field("/app/scripts/testset.csv", description="생성된 테스트셋을 저장할 CSV 파일 경로")

class TestsetGenerateResponse(BaseModel):
    status: str
    message: str
    output_file: str
    sample_preview: List[dict]

class EvaluationRequest(BaseModel):
    testset_file: str = "/app/scripts/testset.csv"
    top_k: int = 5

class EvaluationResponse(BaseModel):
    status: str
    message: str
    results_file: str
    metrics: dict
