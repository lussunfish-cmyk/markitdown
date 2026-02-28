#!/bin/bash

################################################################################
# markitdown FastAPI 서버 실행 관리 스크립트
# 
# 사용법:
#   ./run.sh start          # 서버 시작
#   ./run.sh start --debug  # 디버그 로그 활성화해서 시작
#   ./run.sh stop           # 서버 중지
#   ./run.sh restart        # 서버 재시작
#   ./run.sh restart --debug # 디버그 모드로 재시작
#   ./run.sh status         # 서버 상태 확인
#   ./run.sh logs           # 로그 파일 읽기
#   ./run.sh logs --tail N  # 최근 N줄 로그 보기
################################################################################

set -e

# ============================================================================
# 설정
# ============================================================================

# 스크립트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 프로젝트 디렉토리
PROJECT_DIR="${SCRIPT_DIR}"

# Python 가상환경
VENV_DIR="${PROJECT_DIR}/.venv"
PYTHON="${VENV_DIR}/bin/python"
UVICORN="${VENV_DIR}/bin/uvicorn"

# 로그 디렉토리
LOG_DIR="${PROJECT_DIR}/logs"
PID_FILE="${LOG_DIR}/server.pid"
LOG_FILE="${LOG_DIR}/server.log"
ERROR_LOG_FILE="${LOG_DIR}/server.error.log"

# 서버 설정
APP_MODULE="app.converter:app"
HOST="0.0.0.0"
PORT="8000"
WORKERS="4"

# ============================================================================
# 색상 정의
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 헬퍼 함수
# ============================================================================

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# 로그 디렉토리 생성
setup_logging() {
    if [ ! -d "${LOG_DIR}" ]; then
        mkdir -p "${LOG_DIR}"
        print_info "로그 디렉토리 생성: ${LOG_DIR}"
    fi
    
    # 로그 파일 초기화 (없으면 생성)
    touch "${LOG_FILE}" "${ERROR_LOG_FILE}"
}

# 가상환경 확인
check_venv() {
    if [ ! -f "${PYTHON}" ]; then
        print_error "가상환경을 찾을 수 없습니다: ${VENV_DIR}"
        print_info "다음 명령으로 가상환경을 생성하세요:"
        echo "  python3 -m venv ${VENV_DIR}"
        exit 1
    fi
}

# 프로세스 확인
is_running() {
    if [ -f "${PID_FILE}" ]; then
        local pid=$(cat "${PID_FILE}" 2>/dev/null)
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# 프로세스 상태 출력
print_status() {
    if is_running; then
        local pid=$(cat "${PID_FILE}")
        print_success "서버 실행 중 (PID: ${pid})"
        return 0
    else
        print_warning "서버가 실행되지 않음"
        return 1
    fi
}

# ============================================================================
# 메인 함수
# ============================================================================

start_server() {
    local debug_mode=false
    local log_level="info"
    
    # 옵션 파싱
    if [[ "$1" == "--debug" ]]; then
        debug_mode=true
        log_level="debug"
    fi
    
    print_info "서버 시작 중..."
    
    # 이미 실행 중인지 확인
    if is_running; then
        local pid=$(cat "${PID_FILE}")
        print_warning "서버가 이미 실행 중입니다 (PID: ${pid})"
        return 0
    fi
    
    # 로그 레벨 표시
    if [ "${debug_mode}" = true ]; then
        print_info "디버그 모드 활성화 (로그 레벨: ${log_level})"
    fi
    
    # 로그 파일 정보
    print_info "로그 파일: ${LOG_FILE}"
    print_info "에러 로그: ${ERROR_LOG_FILE}"
    
    # nohup으로 서버 시작
    nohup ${PYTHON} -m uvicorn \
        "${APP_MODULE}" \
        --host "${HOST}" \
        --port "${PORT}" \
        --log-level "${log_level}" \
        --loop asyncio \
        > "${LOG_FILE}" 2> "${ERROR_LOG_FILE}" &
    
    local pid=$!
    echo "${pid}" > "${PID_FILE}"
    
    # 서버 시작 확인 (2초 대기)
    sleep 2
    
    if is_running; then
        print_success "서버 시작 완료 (PID: ${pid})"
        print_info "서버 주소: http://${HOST}:${PORT}"
        print_info "API 문서: http://${HOST}:${PORT}/docs"
        return 0
    else
        print_error "서버 시작 실패"
        print_warning "에러 로그 확인:"
        tail -20 "${ERROR_LOG_FILE}"
        rm -f "${PID_FILE}"
        return 1
    fi
}

stop_server() {
    print_info "서버 중지 중..."
    
    if ! is_running; then
        print_warning "서버가 실행되지 않음"
        rm -f "${PID_FILE}" 2>/dev/null
        return 0
    fi
    
    local pid=$(cat "${PID_FILE}")
    
    # 프로세스 종료 시도 (SIGTERM)
    kill -TERM "${pid}" 2>/dev/null || true
    
    # 최대 10초 대기
    local count=0
    while is_running && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done
    
    # 강제 종료 (SIGKILL)
    if is_running; then
        print_warning "강제 종료 실행 중..."
        kill -KILL "${pid}" 2>/dev/null || true
        sleep 1
    fi
    
    rm -f "${PID_FILE}"
    print_success "서버 중지 완료"
}

restart_server() {
    local debug_mode=""
    if [[ "$1" == "--debug" ]]; then
        debug_mode="--debug"
    fi
    
    print_info "서버 재시작 중..."
    stop_server
    sleep 1
    start_server "${debug_mode}"
}

show_status() {
    print_info "서버 상태:"
    echo ""
    print_status
    
    if [ -f "${PID_FILE}" ]; then
        local pid=$(cat "${PID_FILE}")
        local process_info=$(ps -p "${pid}" -o cmd= 2>/dev/null || echo "N/A")
        print_info "프로세스 정보: ${process_info}"
        
        echo ""
        print_info "최근 로그 (5줄):"
        tail -5 "${LOG_FILE}" 2>/dev/null || echo "  (로그 없음)"
    fi
}

show_logs() {
    local tail_count=50
    
    # 옵션 파싱
    if [[ "$1" == "--tail" && -n "$2" ]]; then
        tail_count=$2
    fi
    
    if [ ! -f "${LOG_FILE}" ]; then
        print_warning "로그 파일이 없습니다: ${LOG_FILE}"
        return 0
    fi
    
    print_info "로그 파일: ${LOG_FILE} (최근 ${tail_count}줄)"
    echo ""
    print_info "=== 표준 로그 ==="
    tail -"${tail_count}" "${LOG_FILE}"
    
    if [ -s "${ERROR_LOG_FILE}" ]; then
        echo ""
        print_warning "=== 에러 로그 ==="
        tail -"${tail_count}" "${ERROR_LOG_FILE}"
    fi
}

show_help() {
    cat << EOF

${BLUE}markitdown FastAPI 서버 관리${NC}

${GREEN}사용법:${NC}
  ./run.sh <command> [options]

${GREEN}명령어:${NC}
  start              서버 시작
  start --debug      디버그 모드로 서버 시작 (상세 로그)
  stop               서버 중지
  restart            서버 재시작
  restart --debug    디버그 모드로 서버 재시작
  status             서버 상태 확인
  logs               전체 로그 표시 (최근 50줄)
  logs --tail N      최근 N줄 로그 표시
  help               이 도움말 표시

${GREEN}예제:${NC}
  # 서버 시작
  ./run.sh start

  # 디버그 로그 활성화해서 시작
  ./run.sh start --debug

  # 서버 상태 확인
  ./run.sh status

  # 최근 100줄 로그 보기
  ./run.sh logs --tail 100

  # 서버 재시작
  ./run.sh restart

${GREEN}파일 경로:${NC}
  프로젝트 디렉토리: ${PROJECT_DIR}
  가상환경: ${VENV_DIR}
  로그 디렉토리: ${LOG_DIR}
  표준 로그: ${LOG_FILE}
  에러 로그: ${ERROR_LOG_FILE}
  PID 파일: ${PID_FILE}

${GREEN}서버 정보:${NC}
  애플리케이션: ${APP_MODULE}
  주소: http://${HOST}:${PORT}
  API 문서: http://${HOST}:${PORT}/docs

EOF
}

# ============================================================================
# 메인 로직
# ============================================================================

main() {
    # 초기화
    setup_logging
    check_venv
    
    # 명령어 처리
    case "${1:-help}" in
        start)
            start_server "$2"
            ;;
        stop)
            stop_server
            ;;
        restart)
            restart_server "$2"
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs "$2" "$3"
            ;;
        help)
            show_help
            ;;
        *)
            print_error "알 수 없는 명령어: ${1}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 스크립트 실행
main "$@"
