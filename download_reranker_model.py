import os
import sys
import shutil
from pathlib import Path
from sentence_transformers import CrossEncoder

# 기본 설정 (환경 변수로 오버라이드 가능)
# config.py를 사용할 수 없는 빌드 단계에서 실행되므로 기본값 하드코딩 필요
DEFAULT_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEFAULT_MODEL_PATH = "/app/models/bge-reranker-v2-m3"

MODEL_NAME = os.getenv("RETRIEVER_RERANKER_MODEL", DEFAULT_MODEL_NAME)
MODEL_PATH = os.getenv("MODEL_DOWNLOAD_PATH", DEFAULT_MODEL_PATH)

def download_model():
    """
    Reranker 모델을 다운로드하고 지정된 경로에 저장합니다.
    Docker 빌드 과정에서 실행되어 오프라인 환경을 위한 모델 파일을 준비합니다.
    """
    print("=" * 60)
    print(f"Starting Model Download Task")
    print(f"- Target Model: {MODEL_NAME}")
    print(f"- Destination:  {MODEL_PATH}")
    print("=" * 60)

    target_dir = Path(MODEL_PATH)

    # 1. 기존 디렉토리 정리
    if target_dir.exists():
        print(f"Warning: Directory {target_dir} already exists. Removing...")
        try:
            shutil.rmtree(target_dir)
        except OSError as e:
            print(f"Error: Failed to remove existing directory: {e}")
            sys.exit(1)
    
    # 2. 모델 다운로드 및 로드
    try:
        print(f"Downloading model '{MODEL_NAME}' from HuggingFace Hub...")
        # CrossEncoder는 내부적으로 sentence-transformers/transformers를 사용하여 다운로드
        model = CrossEncoder(MODEL_NAME)
    except Exception as e:
        print(f"Error: Failed to download model: {e}")
        sys.exit(1)
    
    # 3. 모델 로컬 저장
    try:
        print(f"Saving model to '{target_dir}'...")
        # 부모 디렉토리가 없으면 생성
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(target_dir))
    except Exception as e:
        print(f"Error: Failed to save model to disk: {e}")
        sys.exit(1)
    
    # 4. 결과 검증
    if target_dir.exists():
        files = list(target_dir.iterdir())
        if files:
            print(f"Success! Model saved to {target_dir}")
            print(f"Total files: {len(files)}")
            for file in files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f" - {file.name:<30} ({size_mb:.2f} MB)")
        else:
            print("Error: Directory created but is empty.")
            sys.exit(1)
    else:
        print("Error: Destination directory was not created.")
        sys.exit(1)
    
    print("=" * 60)
    print("Model preparation completed successfully.")

if __name__ == "__main__":
    download_model()
