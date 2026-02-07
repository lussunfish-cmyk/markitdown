// ============================================================================
// 상태 관리
// ============================================================================

const state = {
    selectedFiles: [],
    uploadedFiles: [],
    documents: [],
    history: []
};

// ============================================================================
// DOM 요소
// ============================================================================

const elements = {
    fileInput: document.getElementById('fileInput'),
    uploadBtn: document.getElementById('uploadBtn'),
    fileList: document.getElementById('fileList'),
    uploadProgress: document.getElementById('uploadProgress'),
    refreshDocsBtn: document.getElementById('refreshDocsBtn'),
    documentsList: document.getElementById('documentsList'),
    queryInput: document.getElementById('queryInput'),
    queryBtn: document.getElementById('queryBtn'),
    topK: document.getElementById('topK'),
    includeSources: document.getElementById('includeSources'),
    queryLoading: document.getElementById('queryLoading'),
    answerArea: document.getElementById('answerArea'),
    answerContent: document.getElementById('answerContent'),
    sourcesArea: document.getElementById('sourcesArea'),
    sourcesList: document.getElementById('sourcesList'),
    historyList: document.getElementById('historyList'),
    clearHistoryBtn: document.getElementById('clearHistoryBtn'),
    toast: document.getElementById('toast')
};

// ============================================================================
// API 호출 함수
// ============================================================================

const API = {
    // 파일 변환
    async convertFile(file, autoIndex = true) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('auto_index', autoIndex.toString());
        
        const response = await fetch('/convert', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '변환 실패');
        }
        
        return await response.json();
    },
    
    // 파일 인덱싱
    async indexFile(filePath) {
        const response = await fetch('/index', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: filePath,
                chunk_size: 500,
                chunk_overlap: 50
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '인덱싱 실패');
        }
        
        return await response.json();
    },
    
    // 문서 목록 조회
    async getDocuments() {
        const response = await fetch('/documents');
        
        if (!response.ok) {
            throw new Error('문서 목록 조회 실패');
        }
        
        return await response.json();
    },
    
    // RAG 질의
    async query(question, topK, includeSources) {
        const response = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: question,
                top_k: topK,
                include_sources: includeSources
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '질의 실패');
        }
        
        return await response.json();
    }
};

// ============================================================================
// UI 헬퍼 함수
// ============================================================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showToast(message, type = 'info') {
    elements.toast.textContent = message;
    elements.toast.className = `toast ${type}`;
    elements.toast.classList.remove('hidden');
    
    setTimeout(() => {
        elements.toast.classList.add('hidden');
    }, 3000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function updateFileList() {
    if (state.selectedFiles.length === 0) {
        elements.fileList.innerHTML = '';
        elements.uploadBtn.disabled = true;
        return;
    }
    
    elements.uploadBtn.disabled = false;
    elements.fileList.innerHTML = state.selectedFiles.map((file, index) => `
        <div class="file-item">
            <div>
                <span class="file-name">${file.name}</span>
                <span class="file-size">(${formatFileSize(file.size)})</span>
            </div>
            <button onclick="removeFile(${index})" class="btn btn-small">삭제</button>
        </div>
    `).join('');
}

function updateDocumentsList() {
    if (state.documents.length === 0) {
        elements.documentsList.innerHTML = '<p class="placeholder">아직 인덱싱된 문서가 없습니다.</p>';
        return;
    }
    
    elements.documentsList.innerHTML = state.documents.map(doc => `
        <div class="document-item">
            <div class="doc-name">📄 ${doc.filename}</div>
            <div class="doc-info">${doc.total_chunks}개 청크 | ID: ${doc.id}</div>
        </div>
    `).join('');
}

function updateHistory() {
    if (state.history.length === 0) {
        elements.historyList.innerHTML = '<p class="placeholder">아직 대화 기록이 없습니다.</p>';
        return;
    }
    
    elements.historyList.innerHTML = state.history.map(item => `
        <div class="history-item">
            <div class="history-question">Q: ${item.question}</div>
            <div class="history-answer">A: ${item.answer}</div>
        </div>
    `).join('');
}

// ============================================================================
// 이벤트 핸들러
// ============================================================================

elements.fileInput.addEventListener('change', (e) => {
    state.selectedFiles = Array.from(e.target.files);
    updateFileList();
});

function removeFile(index) {
    state.selectedFiles.splice(index, 1);
    updateFileList();
}

elements.uploadBtn.addEventListener('click', async () => {
    if (state.selectedFiles.length === 0) return;
    
    elements.uploadBtn.disabled = true;
    elements.uploadProgress.classList.remove('hidden');
    elements.uploadProgress.innerHTML = '';
    
    for (const file of state.selectedFiles) {
        const progressId = `progress-${Date.now()}`;
        elements.uploadProgress.innerHTML += `
            <div class="progress-item" id="${progressId}">
                <div class="progress-header">
                    <span>${file.name}</span>
                    <span class="progress-status">처리 중...</span>
                </div>
            </div>
        `;
        
        const progressEl = document.getElementById(progressId);
        
        try {
            // 파일 변환 & 인덱싱 (auto_index=true)
            progressEl.querySelector('.progress-status').textContent = '변환 & 인덱싱 중...';
            const convertResult = await API.convertFile(file, true);
            
            // 성공
            progressEl.classList.add('success');
            progressEl.querySelector('.progress-status').textContent = '완료 ✓';
            progressEl.querySelector('.progress-status').classList.add('success');
            
            state.uploadedFiles.push(file.name);
        } catch (error) {
            // 실패
            progressEl.classList.add('error');
            progressEl.querySelector('.progress-status').textContent = `실패: ${error.message}`;
            progressEl.querySelector('.progress-status').classList.add('error');
            showToast(`${file.name} 처리 실패: ${error.message}`, 'error');
        }
    }
    
    // 완료 후 문서 목록 새로고침
    await loadDocuments();
    showToast('파일 처리 완료!', 'success');
    
    // 리셋
    state.selectedFiles = [];
    elements.fileInput.value = '';
    updateFileList();
    elements.uploadBtn.disabled = true;
});

elements.refreshDocsBtn.addEventListener('click', loadDocuments);

async function loadDocuments() {
    try {
        const result = await API.getDocuments();
        state.documents = result.documents || [];
        updateDocumentsList();
    } catch (error) {
        showToast('문서 목록 로드 실패: ' + error.message, 'error');
    }
}

elements.queryBtn.addEventListener('click', async () => {
    const question = elements.queryInput.value.trim();
    if (!question) {
        showToast('질문을 입력하세요', 'warning');
        return;
    }
    
    if (state.documents.length === 0) {
        showToast('먼저 문서를 업로드하고 인덱싱하세요', 'warning');
        return;
    }
    
    // UI 업데이트
    elements.queryBtn.disabled = true;
    elements.queryLoading.classList.remove('hidden');
    elements.answerArea.classList.add('hidden');
    
    try {
        const topK = parseInt(elements.topK.value);
        const includeSources = elements.includeSources.checked;
        
        const result = await API.query(question, topK, includeSources);
        
        // 답변 표시
        elements.answerContent.textContent = result.answer;
        elements.answerArea.classList.remove('hidden');
        
        // 출처 표시
        if (includeSources && result.sources && result.sources.length > 0) {
            elements.sourcesArea.classList.remove('hidden');
            elements.sourcesList.innerHTML = result.sources.map((source, index) => `
                <div class="source-item">
                    <div class="source-header">
                        <span class="source-file">📄 ${escapeHtml(source.source)} (Chunk ${source.chunk_id})</span>
                        <span class="source-score">유사도: ${(source.similarity_score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="source-content">${escapeHtml(source.content || '내용 없음')}</div>
                </div>
            `).join('');
        } else {
            elements.sourcesArea.classList.add('hidden');
        }
        
        // 히스토리 추가
        state.history.unshift({
            question: question,
            answer: result.answer,
            timestamp: new Date().toLocaleString()
        });
        updateHistory();
        
        // 입력창 초기화
        elements.queryInput.value = '';
        
        showToast('답변이 생성되었습니다', 'success');
    } catch (error) {
        showToast('질의 실패: ' + error.message, 'error');
        elements.answerArea.classList.add('hidden');
    } finally {
        elements.queryLoading.classList.add('hidden');
        elements.queryBtn.disabled = false;
    }
});

// Enter 키로 질문 전송
elements.queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        elements.queryBtn.click();
    }
});

elements.clearHistoryBtn.addEventListener('click', () => {
    if (confirm('대화 기록을 모두 삭제하시겠습니까?')) {
        state.history = [];
        updateHistory();
        showToast('대화 기록이 삭제되었습니다', 'success');
    }
});

// ============================================================================
// 초기화
// ============================================================================

async function init() {
    console.log('MarkItDown RAG Assistant 초기화...');
    await loadDocuments();
    showToast('준비 완료!', 'success');
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', init);
