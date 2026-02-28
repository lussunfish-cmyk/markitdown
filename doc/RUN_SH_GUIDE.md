# run.sh - FastAPI 서버 관리 스크립트

## 개요

`run.sh`는 markitdown FastAPI 서버를 **nohup**을 사용하여 백그라운드에서 실행하고 관리하는 스크립트입니다.

### 주요 기능

✅ **nohup 백그라운드 실행** - 터미널 종료 후에도 계속 실행  
✅ **디버그 모드** - `--debug` 옵션으로 상세 로그 활성화  
✅ **로그 관리** - 표준 로그와 에러 로그 분리 저장  
✅ **프로세스 관리** - 시작/중지/재시작/상태 확인  
✅ **컬러 출력** - 이해하기 쉬운 색상 기반 메시지  

---

## 설치

스크립트는 이미 실행 가능하도록 설정되어 있습니다:

```bash
# 권한 확인 (필요 시)
chmod +x run.sh
```

---

## 기본 사용법

### 1. 서버 시작

```bash
# 일반 모드 시작
./run.sh start

# 디버그 모드로 시작 (상세 로그)
./run.sh start --debug
```

**출력 예:**
```
ℹ 서버 시작 중...
ℹ 로그 파일: ./logs/server.log
ℹ 에러 로그: ./logs/server.error.log
✓ 서버 시작 완료 (PID: 68714)
ℹ 서버 주소: http://0.0.0.0:8000
ℹ API 문서: http://0.0.0.0:8000/docs
```

### 2. 서버 상태 확인

```bash
./run.sh status
```

**출력 예:**
```
ℹ 서버 상태:

✓ 서버 실행 중 (PID: 68714)
ℹ 프로세스 정보: /usr/local/bin/python -m uvicorn...
ℹ 최근 로그 (5줄):
INFO:     Application startup complete.
```

### 3. 로그 확인

```bash
# 최근 50줄 로그 (기본값)
./run.sh logs

# 최근 N줄 로그
./run.sh logs --tail 100

# 모든 로그 (파이프 사용)
./run.sh logs --tail 9999
```

### 4. 서버 중지

```bash
./run.sh stop
```

**종료 프로세스:**
1. SIGTERM 신호로 우아한 종료 시도 (10초)
2. 타임아웃 시 SIGKILL로 강제 종료
3. PID 파일 정리

### 5. 서버 재시작

```bash
# 일반 모드로 재시작
./run.sh restart

# 디버그 모드로 재시작
./run.sh restart --debug
```

### 6. 도움말

```bash
./run.sh help
```

---

## 옵션 상세 설명

### `--debug` 옵션

디버그 로그를 활성화합니다. 상세한 로그 정보가 필요할 때 사용합니다.

```bash
# 디버그 모드로 시작
./run.sh start --debug

# 로그 확인 (DEBUG 레벨 메시지 포함)
./run.sh logs --tail 50
```

**로그 레벨 비교:**

| 모드 | 레벨 | 메시지 |
|-----|------|--------|
| 일반 | INFO | 시작 완료, 요청 처리 등 중요 정보만 |
| 디버그 | DEBUG | 모든 내부 처리 과정 상세 로그 |

### `--tail N` 옵션

최근 N줄의 로그를 표시합니다.

```bash
# 최근 50줄 (기본값)
./run.sh logs

# 최근 200줄
./run.sh logs --tail 200

# 리알타임 모니터링 (위의 다른 터미널)
tail -f logs/server.log
```

---

## 파일 구조

```
markitdown/
├── run.sh                    # 이 스크립트
├── .venv/                    # Python 가상환경
├── app/                      # 애플리케이션
│   └── converter.py          # FastAPI 앱
└── logs/                     # 로그 디렉토리 (자동 생성)
    ├── server.log           # 표준 출력 로그
    ├── server.error.log     # 에러 로그
    └── server.pid           # 실행 중인 프로세스 ID
```

---

## 로그 관리

### 로그 위치

| 파일 | 용도 |
|-----|------|
| `logs/server.log` | 표준 출력 (INFO, DEBUG 등) |
| `logs/server.error.log` | 에러 및 예외 정보 |
| `logs/server.pid` | 실행 중인 프로세스 ID |

### 로그 모니터링

#### 방법 1: 스크립트 사용

```bash
# 최근 로그 확인
./run.sh logs --tail 50

# 계속 갱신되는 로그 보기
tail -f logs/server.log
```

#### 방법 2: 다른 터미널에서 모니터링

```bash
# 터미널 A: 서버 실행
./run.sh start --debug

# 터미널 B: 로그 모니터링
tail -f logs/server.log
```

#### 방법 3: 에러만 확인

```bash
tail -f logs/server.error.log
```

### 로그 정리

로그 파일이 커질 수 있으므로 정기적으로 정리합니다:

```bash
# 로그 디렉토리 초기화
rm -rf logs

# 또는 특정 로그만 정리
rm logs/server.log
rm logs/server.error.log
```

---

## 실제 사용 예제

### 예제 1: 개발 환경에서 디버그 모드 운영

```bash
# 터미널 1: 서버 시작 (디버그 모드)
./run.sh start --debug

# 터미널 2: 로그 모니터링
tail -f logs/server.log

# 터미널 3: API 테스트
curl http://localhost:8000/docs
```

### 예제 2: 여러 번 재시작

```bash
# 설정 변경 후 재시작
./run.sh restart

# 또는 디버그 모드로 재시작
./run.sh restart --debug

# 상태 확인
./run.sh status
```

### 예제 3: 문제 진단

```bash
# 1. 현재 상태 확인
./run.sh status

# 2. 상세 로그 확인
./run.sh logs --tail 100

# 3. 디버그 모드로 재시작
./run.sh stop
./run.sh start --debug

# 4. 문제 재현 및 로그 검토
./run.sh logs --tail 100

# 5. 일반 모드로 복구
./run.sh restart
```

---

## 트러블슈팅

### 문제 1: "가상환경을 찾을 수 없습니다"

```
✗ 가상환경을 찾을 수 없습니다: ./.venv
```

**해결책:**

```bash
# 가상환경 생성
python3 -m venv .venv

# 의존성 설치
.venv/bin/pip install -r requirements.txt

# 서버 시작
./run.sh start
```

### 문제 2: "포트가 이미 사용 중입니다"

```
ERROR: [Errno 48] Address already in use
```

**해결책:**

```bash
# 포트 사용 프로세스 찾기
lsof -i :8000

# 프로세스 강제 종료 (필요 시)
kill -9 <PID>

# 또는 run.sh로 정상 종료
./run.sh stop
```

### 문제 3: 서버가 시작되지 않음

```bash
# 1. 에러 로그 확인
./run.sh logs --tail 50

# 2. 상세 정보 보기
cat logs/server.error.log

# 3. 디버그 모드로 재시작
./run.sh restart --debug
```

### 문제 4: 좀비 프로세스

```bash
# PID 파일 정리
rm logs/server.pid

# 수동으로 프로세스 종료
kill -9 $(lsof -t -i:8000)

# 서버 재시작
./run.sh start
```

---

## 스크립트 내부 구조

### 주요 함수

| 함수 | 용도 |
|-----|------|
| `start_server()` | nohup으로 서버 시작 |
| `stop_server()` | 서버 종료 (SIGTERM → SIGKILL) |
| `restart_server()` | 서버 재시작 |
| `is_running()` | 실행 여부 확인 |
| `show_status()` | 상태 정보 출력 |
| `show_logs()` | 로그 파일 표시 |

### nohup 사용 이유

```bash
# nohup 없이: 터미널 종료 시 서버도 종료됨
python -m uvicorn app.converter:app ...

# nohup 사용: 터미널 종료 후에도 계속 실행
nohup python -m uvicorn app.converter:app > server.log 2> server.error.log &
```

### 로그 리디렉션

```bash
# STDOUT(표준 출력) → server.log
> server.log

# STDERR(표준 에러) → server.error.log  
2> server.error.log

# 백그라운드 실행
&
```

---

## 고급 사용법

### 1. PID로 프로세스 상태 확인

```bash
# PID 파일에서 프로세스 ID 읽기
pid=$(cat logs/server.pid)

# 프로세스 정보 확인
ps aux | grep $pid

# 또는
ps -p $pid -o cmd=
```

### 2. 구독형 로그 모니터링

```bash
# 커스텀 필터링
./run.sh logs --tail 100 | grep ERROR

# 또는
tail -f logs/server.log | grep -i error
```

### 3. 로그 백업

```bash
# 현재 로그 백업
cp logs/server.log logs/server.log.$(date +%Y%m%d_%H%M%S)

# 로그 초기화
:> logs/server.log
:> logs/server.error.log
```

### 4. 크론 작업으로 정기 실행

```bash
# crontab 편집
crontab -e

# 매일 자정에 서버 재시작
0 0 * * * cd /path/to/markitdown && ./run.sh restart

# 매시간 상태 확인 (실패 시 알림)
0 * * * * cd /path/to/markitdown && ./run.sh status || ./run.sh start
```

---

## 보안 고려사항

### 1. 파일 권한

```bash
# run.sh: 실행 권한만 필요
chmod 755 run.sh

# 로그 파일: 읽기만 필요 (민감 정보 포함 가능)
chmod 644 logs/server.log
```

### 2. 포트 보안

```bash
# 방화벽에서 포트 제한 (필요 시)
sudo ufw allow 8000/tcp

# 또는 SSH 터널만 사용
ssh -L 8000:localhost:8000 user@remote-host
```

### 3. 로그 보안

```bash
# 민감한 정보가 로그에 남지 않도록 주의
# (패스워드, API 키 등)

# 로그 로테이션 설정 (선택사항)
# logrotate 설정으로 자동 관리
```

---

## 성능 모니터링

```bash
# 메모리 사용량 확인
ps aux | grep uvicorn | grep -v grep

# 포트 사용량
netstat -tlnp | grep 8000

# 실시간 모니터링
top -p $(cat logs/server.pid)
```

---

## 참고사항

- 스크립트는 **bash** 셸을 사용합니다
- **macOS**와 **Linux**에서 모두 작동합니다
- 가상환경은 `.venv` 디렉토리를 기대합니다
- 로그는 `logs/` 디렉토리에 저장됩니다 (자동 생성)
