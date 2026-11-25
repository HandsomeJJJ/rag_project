<script setup>
import { computed, nextTick, onMounted, ref } from 'vue';
import DOMPurify from 'dompurify';
import { marked } from 'marked';

const apiBase =
  import.meta.env.VITE_API_BASE_URL ||
  (typeof window !== 'undefined'
    ? `http://${window.location.hostname}:8000`
    : 'http://localhost:8000');
const sessions = ref([]);
const activeId = ref(null);
const messages = ref([]);
const question = ref('');
const busy = ref(false);
const busyStage = ref('');
const error = ref('');
const notice = ref('');
const fileInput = ref(null);
const chatPane = ref(null);
const searchText = ref('');
const editingSessionId = ref(null);
const editingTitle = ref('');
const openMenuSessionId = ref(null);
const copiedMessageIndex = ref(null);
const activeController = ref(null);
const sourceFilter = ref('');
const documents = ref([]);
const documentsOpen = ref(false);
const docUploadInput = ref(null);
const uploadJob = ref(null);
const feedbackState = ref({});
const expandedSource = ref({});
const shareMode = ref(false);
const shareToken = ref('');
const shareTitle = ref('');
const shareSourceFilter = ref('');
let sessionRequestToken = 0;

const activeTitle = computed(() => {
  if (shareMode.value) return shareTitle.value || '分享的会话';
  return (
    sessions.value.find((session) => session.session_id === activeId.value)?.title ||
    '新会话'
  );
});
const availableSources = computed(() => documents.value.map((doc) => doc.source));
const visibleSessions = computed(() => {
  const query = searchText.value.trim().toLocaleLowerCase();
  if (!query) return sessions.value;
  return sessions.value.filter((session) =>
    session.title.toLocaleLowerCase().includes(query)
  );
});

function showNotice(text) {
  notice.value = text;
  window.setTimeout(() => {
    if (notice.value === text) notice.value = '';
  }, 4000);
}

async function apiFetch(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (typeof options.body === 'string' && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }
  return fetch(`${apiBase}${path}`, {
    credentials: 'include',
    ...options,
    headers,
  });
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

async function loadSessions() {
  const response = await apiFetch('/sessions');
  if (!response.ok) throw new Error('会话列表加载失败');
  sessions.value = await response.json();
}

async function loadDocuments() {
  const response = await apiFetch('/documents');
  if (!response.ok) throw new Error('知识库列表加载失败');
  const data = await readJson(response);
  documents.value = data.documents || [];
  if (
    sourceFilter.value &&
    !documents.value.some((doc) => doc.source === sourceFilter.value)
  ) {
    sourceFilter.value = '';
  }
}

async function fetchSession(sessionId) {
  const response = await apiFetch(`/sessions/${encodeURIComponent(sessionId)}`);
  const data = await readJson(response);
  if (!response.ok) throw new Error(data.detail || '会话加载失败');
  return data;
}

async function selectSession(session) {
  if (editingSessionId.value) return;
  const token = ++sessionRequestToken;
  if (busy.value) stopGeneration();
  activeId.value = session.session_id;
  messages.value = session.messages || [];
  sourceFilter.value = session.source_filter || '';
  if (
    documents.value.length &&
    sourceFilter.value &&
    !availableSources.value.includes(sourceFilter.value)
  ) {
    sourceFilter.value = '';
  }
  error.value = '';
  openMenuSessionId.value = null;
  expandedSource.value = {};
  feedbackState.value = {};
  try {
    const detail = await fetchSession(session.session_id);
    if (token !== sessionRequestToken) return;
    session.messages = detail.messages || [];
    session.source_filter = detail.source_filter || '';
    messages.value = session.messages;
    sourceFilter.value = session.source_filter;
  } catch (err) {
    if (token === sessionRequestToken) {
      error.value = err.message || '会话加载失败';
    }
  }
  await scrollBottom();
}

function newChat() {
  if (busy.value) stopGeneration();
  activeId.value = null;
  messages.value = [];
  error.value = '';
  notice.value = '';
  openMenuSessionId.value = null;
  editingSessionId.value = null;
  expandedSource.value = {};
  feedbackState.value = {};
}

function toggleSessionMenu(sessionId) {
  openMenuSessionId.value =
    openMenuSessionId.value === sessionId ? null : sessionId;
}

function beginRename(session) {
  editingSessionId.value = session.session_id;
  editingTitle.value = session.title;
  openMenuSessionId.value = null;
  nextTick(() => {
    document.querySelector(`[data-session-title="${session.session_id}"]`)?.focus();
  });
}

function cancelRename() {
  editingSessionId.value = null;
  editingTitle.value = '';
}

async function saveRename(session) {
  const title = editingTitle.value.trim();
  if (!title) return;
  try {
    const response = await apiFetch(
      `/sessions/${encodeURIComponent(session.session_id)}`,
      {
        method: 'PATCH',
        body: JSON.stringify({ title }),
      }
    );
    const result = await readJson(response);
    if (!response.ok) throw new Error(result.detail || '重命名失败');
    session.title = result.title;
    cancelRename();
    await loadSessions();
  } catch (err) {
    error.value = err.message || '重命名失败';
  }
}

async function togglePin(session) {
  try {
    const response = await apiFetch(
      `/sessions/${encodeURIComponent(session.session_id)}/pin`,
      {
        method: 'POST',
        body: JSON.stringify({ pinned: !session.pinned }),
      }
    );
    const result = await readJson(response);
    if (!response.ok) throw new Error(result.detail || '置顶操作失败');
    session.pinned = result.pinned;
    openMenuSessionId.value = null;
    await loadSessions();
  } catch (err) {
    error.value = err.message || '置顶操作失败';
  }
}

async function deleteSession(session) {
  if (!window.confirm(`确定删除会话“${session.title}”吗？此操作无法撤销。`))
    return;
  try {
    const response = await apiFetch(
      `/sessions/${encodeURIComponent(session.session_id)}`,
      { method: 'DELETE' }
    );
    const result = await readJson(response);
    if (!response.ok) throw new Error(result?.detail || '删除失败');
    sessions.value = sessions.value.filter(
      (item) => item.session_id !== session.session_id
    );
    openMenuSessionId.value = null;
    if (activeId.value === session.session_id) newChat();
  } catch (err) {
    error.value = err.message || '删除失败';
  }
}

async function shareSession(session) {
  openMenuSessionId.value = null;
  try {
    const response = await apiFetch(
      `/sessions/${encodeURIComponent(session.session_id)}/share`,
      { method: 'POST' }
    );
    const result = await readJson(response);
    if (!response.ok) throw new Error(result.detail || '分享失败');
    const link = `${window.location.origin}${window.location.pathname}?share=${result.token}`;
    try {
      await navigator.clipboard.writeText(link);
      showNotice('分享链接已复制到剪贴板');
    } catch {
      window.prompt('复制分享链接', link);
    }
  } catch (err) {
    error.value = err.message || '分享失败';
  }
}

function exportMarkdown(session) {
  openMenuSessionId.value = null;
  window.open(
    `${apiBase}/sessions/${encodeURIComponent(session.session_id)}/export`,
    '_blank'
  );
}

function escapeHtml(text) {
  return String(text || '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function sourceLabel(source) {
  return [source.source, source.article_no, source.chapter]
    .filter(Boolean)
    .join(' · ');
}

function sourceMeta(source) {
  return [
    source.source,
    source.part,
    source.chapter,
    source.section,
    source.article_no && `起始：${source.article_no}`,
    source.article_end && `覆盖至：${source.article_end}`,
  ]
    .filter(Boolean)
    .join(' · ');
}

function highlightSource(source) {
  const text = escapeHtml(source.content || '');
  const label = escapeHtml(source.article_no || '');
  if (!label) return text;
  return text.replaceAll(label, `<mark>${label}</mark>`);
}

function toggleSource(messageIndex, sourceIndex) {
  const current = expandedSource.value[messageIndex];
  expandedSource.value = {
    ...expandedSource.value,
    [messageIndex]: current === sourceIndex ? -1 : sourceIndex,
  };
}

async function exportPdf(session) {
  openMenuSessionId.value = null;
  const printWindow = window.open('', '_blank');
  if (!printWindow) {
    error.value = '浏览器阻止了弹窗，请允许后重试';
    return;
  }
  let sessionMessages = session.messages;
  if (!sessionMessages) {
    printWindow.document.write(
      '<p style="font-family:sans-serif;padding:24px;color:#3f6547">正在准备导出…</p>'
    );
    try {
      const detail = await fetchSession(session.session_id);
      sessionMessages = detail.messages || [];
      session.messages = sessionMessages;
    } catch (err) {
      printWindow.close();
      error.value = err.message || '导出失败';
      return;
    }
  }
  const body = (sessionMessages || [])
    .map((message) => {
      const content =
        message.role === 'user'
          ? `<p class="user-text">${escapeHtml(message.content)}</p>`
          : `<div class="markdown-body">${renderMarkdown(message.content)}</div>`;
      const sources = (message.sources || [])
        .map(
          (source) =>
            `<li>[${source.index}] ${escapeHtml(sourceLabel(source))}</li>`
        )
        .join('');
      return `<section class="message ${message.role}">
        <div class="role">${message.role === 'user' ? '你' : '法律助手'}</div>
        ${content}
        ${sources ? `<div class="sources"><strong>参考来源</strong><ul>${sources}</ul></div>` : ''}
      </section>`;
    })
    .join('');
  printWindow.document.write(`<!doctype html>
  <html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <title>${escapeHtml(session.title)}</title>
    <style>
      body { font-family: "Noto Sans SC", "Microsoft YaHei", sans-serif; color: #1c3524; margin: 36px; line-height: 1.8; }
      h1 { font-size: 22px; border-bottom: 2px solid #1d483b; padding-bottom: 10px; }
      .message { margin: 22px 0; page-break-inside: avoid; }
      .role { font-weight: 700; color: #1d483b; margin-bottom: 6px; }
      .user-text { background: #eef6ef; padding: 10px 12px; border-radius: 8px; white-space: pre-wrap; }
      .markdown-body h3 { font-size: 15px; margin: 12px 0 6px; }
      .sources { font-size: 12px; color: #526f58; margin-top: 8px; }
      .sources ul { padding-left: 20px; }
    </style>
  </head>
  <body>
    <h1>${escapeHtml(session.title)}</h1>
    ${body}
  </body>
  </html>`);
  printWindow.document.close();
  printWindow.setTimeout(() => printWindow.print(), 300);
}

async function scrollBottom() {
  await nextTick();
  if (chatPane.value) chatPane.value.scrollTop = chatPane.value.scrollHeight;
}

async function send() {
  const text = question.value.trim();
  if (!text || busy.value) return;
  question.value = '';
  return streamQuestion(text);
}

async function streamQuestion(text, replaceIndex = null) {
  if (!text || busy.value) return;
  error.value = '';
  let answer;
  let previousAnswer = '';
  if (replaceIndex === null) {
    messages.value.push(
      { role: 'user', content: text },
      { role: 'assistant', content: '', sources: [], citationWarning: '' }
    );
    answer = messages.value[messages.value.length - 1];
  } else {
    answer = messages.value[replaceIndex];
    if (!answer || answer.role !== 'assistant') return;
    previousAnswer = answer.content;
    answer.content = '';
    answer.sources = [];
    answer.citationWarning = '';
  }
  busy.value = true;
  busyStage.value = '正在准备…';
  let completed = false;
  const controller = new AbortController();
  activeController.value = controller;
  await scrollBottom();
  try {
    const response = await apiFetch('/chat', {
      method: 'POST',
      signal: controller.signal,
      body: JSON.stringify({
        question: text,
        session_id: activeId.value,
        source_filter: sourceFilter.value,
        ...(replaceIndex === null
          ? {}
          : { regenerate_message_index: replaceIndex }),
      }),
    });
    if (!response.ok || !response.body) {
      const detail = await readJson(response);
      throw new Error(detail.detail || `请求失败 (${response.status})`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      const blocks = pending.split('\n\n');
      pending = blocks.pop() || '';
      let tokenBuffer = '';
      for (const block of blocks) {
        let eventName = 'message';
        const dataLines = [];
        for (const line of block.split('\n')) {
          if (line.startsWith('event:')) eventName = line.slice(6).trim();
          else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
        }
        if (!dataLines.length) continue;
        let payload;
        try {
          payload = JSON.parse(dataLines.join('\n'));
        } catch {
          continue;
        }
        if (eventName === 'session' && payload.session_id) {
          activeId.value = payload.session_id;
        } else if (eventName === 'status') {
          busyStage.value = payload.text || '正在处理…';
        } else if (eventName === 'sources') {
          answer.sources = payload.sources || [];
          answer.retrievedQuery = payload.query || '';
        } else if (eventName === 'token') {
          tokenBuffer += payload.content || '';
          busyStage.value = '';
        } else if (eventName === 'done') {
          if (tokenBuffer) {
            answer.content += tokenBuffer;
            tokenBuffer = '';
          }
          answer.sources = payload.sources || answer.sources || [];
          answer.citationWarning = payload.citation_warning || '';
          busyStage.value = '';
        } else if (eventName === 'error') {
          throw new Error(payload.detail || '回答生成失败');
        }
      }
      if (tokenBuffer) answer.content += tokenBuffer;
      await scrollBottom();
    }
    completed = true;
    const completedSessionId = activeId.value;
    await loadSessions();
    if (completedSessionId && activeId.value === completedSessionId) {
      const detail = await fetchSession(completedSessionId);
      if (activeId.value === completedSessionId) {
        const refreshed = sessions.value.find(
          (session) => session.session_id === completedSessionId
        );
        if (refreshed) {
          refreshed.messages = detail.messages || [];
          refreshed.source_filter = detail.source_filter || '';
        }
        messages.value = detail.messages || [];
        sourceFilter.value = detail.source_filter || '';
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') return;
    controller.abort();
    error.value = err.message || '服务连接失败';
    if (replaceIndex !== null) answer.content = previousAnswer;
    else if (!answer.content) messages.value.pop();
  } finally {
    busy.value = false;
    busyStage.value = '';
    activeController.value = null;
    if (completed) feedbackState.value = {};
  }
}

function stopGeneration() {
  activeController.value?.abort();
}

function previousQuestionFor(messageIndex) {
  for (let index = messageIndex - 1; index >= 0; index -= 1) {
    if (messages.value[index].role === 'user') return messages.value[index].content;
  }
  return '';
}

function regenerate(messageIndex) {
  const text = previousQuestionFor(messageIndex);
  if (!text) {
    error.value = '找不到该回答对应的问题';
    return;
  }
  streamQuestion(text, messageIndex);
}

async function copyAnswer(messageIndex, text) {
  try {
    await navigator.clipboard.writeText(text || '');
    copiedMessageIndex.value = messageIndex;
    window.setTimeout(() => {
      if (copiedMessageIndex.value === messageIndex)
        copiedMessageIndex.value = null;
    }, 1500);
  } catch {
    error.value = '复制失败，请检查浏览器剪贴板权限';
  }
}

async function submitFeedback(messageIndex, rating) {
  const message = messages.value[messageIndex];
  if (!message || !activeId.value) return;
  let comment = '';
  if (rating === 'citation_wrong') {
    comment = window.prompt('请简单描述引用问题（可留空）：') || '';
  }
  try {
    const response = await apiFetch('/feedback', {
      method: 'POST',
      body: JSON.stringify({
        session_id: activeId.value,
        message_index: messageIndex,
        rating,
        question: previousQuestionFor(messageIndex),
        answer: message.content,
        sources: message.sources || [],
        comment,
      }),
    });
    const result = await readJson(response);
    if (!response.ok) throw new Error(result.detail || '反馈提交失败');
    feedbackState.value = { ...feedbackState.value, [messageIndex]: rating };
    showNotice(rating === 'up' ? '感谢反馈，已记录为有帮助' : '已记录反馈，我们会持续改进');
  } catch (err) {
    error.value = err.message || '反馈提交失败';
  }
}

async function persistSourceFilter() {
  if (!activeId.value) return;
  try {
    await apiFetch(
      `/sessions/${encodeURIComponent(activeId.value)}/source-filter`,
      {
        method: 'POST',
        body: JSON.stringify({ source_filter: sourceFilter.value }),
      }
    );
    const session = sessions.value.find((item) => item.session_id === activeId.value);
    if (session) session.source_filter = sourceFilter.value;
  } catch (err) {
    error.value = err.message || '检索范围保存失败';
  }
}

async function uploadFile(event) {
  const file = event.target.files?.[0];
  event.target.value = '';
  if (!file) return;
  documentsOpen.value = true;
  try {
    await uploadSingle(file);
    await loadDocuments();
  } catch (err) {
    error.value = err.message || '文件上传失败';
  }
}

async function uploadSingle(file) {
  const form = new FormData();
  form.append('file', file);
  const response = await apiFetch('/upload', { method: 'POST', body: form });
  const result = await readJson(response);
  if (!response.ok) throw new Error(result.detail || '上传失败');
  return pollJob(result.job_id, `上传 ${file.name}`);
}

async function pollJob(jobId, label) {
  uploadJob.value = { label, progress: 0, message: '已提交任务…' };
  for (;;) {
    await new Promise((resolve) => window.setTimeout(resolve, 500));
    const response = await apiFetch(`/jobs/${encodeURIComponent(jobId)}`);
    const job = await readJson(response);
    if (!response.ok) throw new Error(job.detail || '任务状态查询失败');
    uploadJob.value = {
      label,
      progress: job.progress || 0,
      message: job.message || job.stage || '处理中…',
    };
    if (job.status === 'done') {
      showNotice(job.result || `${label} 完成`);
      return job;
    }
    if (job.status === 'error') {
      throw new Error(job.error || job.message || '任务失败');
    }
  }
}

function openDocuments() {
  documentsOpen.value = true;
  notice.value = '';
  error.value = '';
  loadDocuments().catch((err) => {
    error.value = err.message || '知识库列表加载失败';
  });
}

async function handleDocUpload(event) {
  const files = Array.from(event.target.files || []);
  event.target.value = '';
  if (!files.length) return;
  try {
    for (const file of files) {
      await uploadSingle(file);
    }
    await loadDocuments();
  } catch (err) {
    error.value = err.message || '上传失败';
  }
}

async function deleteDocument(doc) {
  if (!window.confirm(`确定从知识库删除“${doc.source}”吗？`)) return;
  try {
    const response = await apiFetch(
      `/documents/${encodeURIComponent(doc.source)}`,
      { method: 'DELETE' }
    );
    const result = await readJson(response);
    if (!response.ok) throw new Error(result.detail || '删除失败');
    if (sourceFilter.value === doc.source) {
      sourceFilter.value = '';
      await persistSourceFilter();
    }
    showNotice(result.result || '删除完成');
    await loadDocuments();
  } catch (err) {
    error.value = err.message || '删除失败';
  }
}

async function rebuildIndex() {
  if (!window.confirm('将清空现有向量索引并重新入库 data/ 目录，确定继续？')) return;
  try {
    const response = await apiFetch('/documents/rebuild', { method: 'POST' });
    const result = await readJson(response);
    if (!response.ok) throw new Error(result.detail || '重建任务创建失败');
    await pollJob(result.job_id, '重建索引');
    await loadDocuments();
  } catch (err) {
    error.value = err.message || '重建索引失败';
  }
}

async function loadSharedSession(token) {
  const response = await apiFetch(`/share/${encodeURIComponent(token)}`);
  const data = await readJson(response);
  if (!response.ok) throw new Error(data.detail || '分享链接无效');
  shareMode.value = true;
  shareToken.value = token;
  shareTitle.value = data.title || '分享的会话';
  shareSourceFilter.value = data.source_filter || '';
  messages.value = data.messages || [];
}

function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text || ''));
}

onMounted(async () => {
  const token = new URLSearchParams(window.location.search).get('share');
  if (token) {
    try {
      await loadSharedSession(token);
    } catch (err) {
      error.value = err.message || '分享链接加载失败';
    }
    return;
  }
  try {
    await loadSessions();
    await loadDocuments().catch(() => {
      // 文档列表失败不影响会话浏览，检索范围保持“全部知识库”
    });
    if (sessions.value.length) await selectSession(sessions.value[0]);
  } catch (err) {
    error.value = `${err.message}，请确认 FastAPI 已启动`;
  }
});
</script>

<template>
  <!-- 页面最外层容器 -->
  <div :class="['layout', { shared: shareMode }]">
    <!-- ========== 左侧边栏 ========== -->
    <aside v-if="!shareMode" class="sidebar">
      <!-- 品牌标题：⚖ 法知 智能法律问答 -->
      <div class="brand">
        <span class="brand-icon">⚖</span>
        <div>
          <strong>打工人</strong>
          <small>智能法律问答</small>
        </div>
      </div>

      <!-- 新建会话按钮，点击调用 newChat() -->
      <button class="new-chat" @click="newChat">
        <span>＋</span> 新建会话
      </button>

      <label class="session-search">
        <span>⌕</span>
        <input v-model="searchText" type="search" placeholder="搜索会话" />
      </label>
      <div class="list-title">最近会话</div>
      <nav class="session-list">
        <!-- 循环渲染会话列表：遍历 sessions数组 -->
        <div
          v-for="session in visibleSessions"
          :key="session.session_id"
          :class="['session-entry', { active: activeId === session.session_id }]"
          @click="selectSession(session)"
        >
          <span class="session-mark">{{ session.pinned ? '📌' : '◷' }}</span>
          <input
            v-if="editingSessionId === session.session_id"
            v-model="editingTitle"
            class="session-rename-input"
            :data-session-title="session.session_id"
            maxlength="80"
            @click.stop
            @keydown.enter.prevent.stop="saveRename(session)"
            @keydown.esc.prevent.stop="cancelRename"
          />
          <span v-else class="session-name">{{ session.title }}</span>
          <button
            v-if="editingSessionId !== session.session_id"
            class="session-menu-trigger"
            type="button"
            aria-label="会话操作"
            @click.stop="toggleSessionMenu(session.session_id)"
          >
            ···
          </button>
          <div
            v-if="openMenuSessionId === session.session_id"
            class="session-menu"
            @click.stop
          >
            <button type="button" @click="beginRename(session)">重命名</button>
            <button type="button" @click="togglePin(session)">
              {{ session.pinned ? '取消置顶' : '置顶' }}
            </button>
            <button type="button" @click="shareSession(session)">分享</button>
            <button type="button" @click="exportMarkdown(session)">
              导出 Markdown
            </button>
            <button type="button" @click="exportPdf(session)">导出 PDF</button>
            <button class="danger-action" type="button" @click="deleteSession(session)">
              删除
            </button>
          </div>
        </div>
        <!-- 如果会话数组为空，显示提示文字 -->
        <div v-if="!visibleSessions.length" class="empty-list">
          {{ searchText ? '没有匹配的会话' : '开始一个新问题' }}
        </div>
      </nav>

      <!-- 侧边栏底部状态栏 -->
      <div class="sidebar-bottom">
        <span class="status-dot"></span>
        <span>本地法律知识库</span>
        <button class="doc-manage-btn" type="button" @click="openDocuments">
          知识库管理
        </button>
      </div>
    </aside>

    <!-- ========== 右侧主区域 ========== -->
    <main class="main">
      <!-- 顶部导航栏 -->
      <header class="topbar">
        <div class="crumb">
          {{ shareMode ? '分享会话' : '法律助手' }}
          <span>/</span> {{ activeTitle }}
        </div>
        <div class="topbar-actions">
          <button
            class="mobile-doc-btn"
            type="button"
            title="知识库管理"
            @click="openDocuments"
          >
            📚
          </button>
          <label v-if="!shareMode" class="filter-select">
            <span>检索范围</span>
            <select v-model="sourceFilter" @change="persistSourceFilter">
              <option value="">全部知识库</option>
              <option v-for="source in availableSources" :key="source" :value="source">
                {{ source }}
              </option>
            </select>
          </label>
          <span v-else class="filter-static">
            范围：{{ shareSourceFilter || '全部知识库' }}
          </span>
          <div class="model-badge"><i></i> RAG 已连接</div>
        </div>
      </header>

      <div v-if="shareMode" class="share-banner">
        这是一个只读分享的会话
        <a href="/">返回我的会话</a>
      </div>
      <div v-if="shareMode && error" class="error-banner standalone">
        {{ error }}
      </div>

      <div class="conversation-shell">
        <!-- 聊天面板容器，ref用于js拿到DOM，自动滚动到底部 -->
        <section ref="chatPane" class="chat-pane">
          <!-- 没有消息的时候：欢迎首页 + 推荐问题按钮 -->
          <div v-if="!messages.length" class="welcome">
            <div class="welcome-symbol">⚖</div>
            <h1>你好，有什么法律问题？</h1>
            <p>基于本地法律知识库检索，并在需要时调用外部工具辅助分析。</p>
            <div v-if="!shareMode" class="suggestions">
              <!-- 快捷提问按钮：点击直接给question赋值 -->
              <button
                @click="question = '试用期被辞退，用人单位需要支付经济补偿吗？'"
              >
                <b>劳动关系</b>
                <span>试用期被辞退，是否有经济补偿？</span>
              </button>
              <button @click="question = '学校处分学生时，应当遵循哪些程序？'">
                <b>教育权益</b>
                <span>学校作出处分时应遵循哪些程序？</span>
              </button>
            </div>
          </div>

          <!-- 有聊天消息时，渲染消息列表 -->
          <div v-else class="messages">
            <!-- 遍历messages消息数组，渲染每一行对话 -->
            <article
              v-for="(message, index) in messages"
              :key="index"
              :class="['message-row', message.role]"
            >
              <!-- 助手消息：左侧头像 -->
              <div
                v-if="message.role === 'assistant'"
                class="avatar assistant-avatar"
              >
                ✨
              </div>

              <div :class="['message-content', message.role]">
                <div class="role-label">
                  {{ message.role === 'user' ? '你' : '法律助手' }}
                </div>
                <!-- 用户消息：直接纯文本渲染 -->
                <div v-if="message.role === 'user'" class="user-text">
                  {{ message.content }}
                </div>
                <!-- 助手消息：markdown渲染，v-html转成HTML -->
                <div
                  v-else
                  class="markdown-body"
                  v-html="renderMarkdown(message.content)"
                ></div>

                <div
                  v-if="busy && index === messages.length - 1"
                  class="status-pill"
                >
                  <span class="status-pulse"></span>
                  {{ busyStage || (message.content ? '正在生成…' : '正在分析…') }}
                </div>

                <div v-if="message.citationWarning" class="citation-warning">
                  ⚠️ {{ message.citationWarning }}
                </div>

                <div
                  v-if="message.role === 'assistant' && message.sources?.length"
                  class="source-block"
                >
                  <div class="source-block-title">
                    参考来源（{{ message.sources.length }}）
                  </div>
                  <div class="source-chips">
                    <button
                      v-for="(source, sourceIndex) in message.sources"
                      :key="sourceIndex"
                      type="button"
                      :class="[
                        'source-chip',
                        { active: expandedSource[index] === sourceIndex },
                      ]"
                      @click="toggleSource(index, sourceIndex)"
                    >
                      [{{ source.index || sourceIndex + 1 }}]
                      {{ sourceLabel(source) || '未命名来源' }}
                    </button>
                  </div>
                  <div
                    v-if="expandedSource[index] >= 0 && message.sources[expandedSource[index]]"
                    class="source-detail"
                  >
                    <div class="source-meta">
                      {{ sourceMeta(message.sources[expandedSource[index]]) }}
                    </div>
                    <pre v-html="highlightSource(message.sources[expandedSource[index]])"></pre>
                  </div>
                </div>

                <div
                  v-if="message.role === 'assistant' && message.content && !(busy && index === messages.length - 1)"
                  class="message-actions"
                >
                  <button type="button" @click="copyAnswer(index, message.content)">
                    {{ copiedMessageIndex === index ? '已复制' : '复制' }}
                  </button>
                  <button
                    v-if="!shareMode"
                    type="button"
                    @click="regenerate(index)"
                  >
                    重新生成
                  </button>
                  <template v-if="!shareMode">
                    <span class="action-divider"></span>
                    <button
                      type="button"
                      :class="{ 'feedback-active': feedbackState[index] === 'up' }"
                      title="有帮助"
                      @click="submitFeedback(index, 'up')"
                    >
                      👍
                    </button>
                    <button
                      type="button"
                      :class="{ 'feedback-active': feedbackState[index] === 'down' }"
                      title="没帮助"
                      @click="submitFeedback(index, 'down')"
                    >
                      👎
                    </button>
                    <button
                      type="button"
                      :class="{ 'feedback-active': feedbackState[index] === 'citation_wrong' }"
                      @click="submitFeedback(index, 'citation_wrong')"
                    >
                      引用有误
                    </button>
                  </template>
                </div>
              </div>
              <!-- 用户消息：右侧头像 -->
              <div v-if="message.role === 'user'" class="avatar user-avatar">
                🙂
              </div>
            </article>
            <!-- 全局错误提示条 -->
            <div v-if="error" class="error-banner">{{ error }}</div>
            <div v-if="notice" class="notice-banner">{{ notice }}</div>
          </div>
        </section>

        <!-- 底部输入区域 -->
        <footer v-if="!shareMode" class="composer-wrap">
          <!-- 空会话时单独展示错误 -->
          <div v-if="error && !messages.length" class="error-banner standalone">
            {{ error }}
          </div>
          <div v-if="notice && !messages.length" class="notice-banner standalone">
            {{ notice }}
          </div>
          <div class="composer">
            <!-- 文本输入框 -->
            <textarea
              v-model="question"
              rows="1"
              placeholder="输入法律问题…"
              :disabled="busy"
              @keydown.enter.exact.prevent="send"
            ></textarea>
            <div class="composer-actions">
              <div class="left-actions">
                <!-- 隐藏的file输入框，点击按钮触发文件选择 -->
                <input
                  ref="fileInput"
                  type="file"
                  accept=".txt,.md,.pdf,.docx"
                  hidden
                  @change="uploadFile"
                />
                <button
                  class="upload-btn"
                  title="上传知识文件"
                  @click="fileInput.click()"
                >
                  <span>＋</span> 上传知识文件
                </button>
                <span class="hint">Enter 发送 · Shift + Enter 换行</span>
              </div>
              <!-- 回答生成中切换为客户端停止按钮 -->
              <button
                :class="['send-btn', { 'stop-btn': busy }]"
                :disabled="!busy && !question.trim()"
                @click="busy ? stopGeneration() : send()"
              >
                {{ busy ? '停止' : '↑' }}
              </button>
            </div>
          </div>
          <!-- 免责声明 -->
          <div class="disclaimer">
            法律助手提供的信息仅供学习参考，不构成正式法律意见。
          </div>
        </footer>
      </div>
    </main>

    <!-- 知识库管理抽屉 -->
    <div
      v-if="documentsOpen"
      class="drawer-backdrop"
      @click.self="documentsOpen = false"
    >
      <aside class="doc-drawer">
        <header class="doc-drawer-header">
          <div>
            <strong>知识库管理</strong>
            <small>上传、删除文档或重建索引</small>
          </div>
          <button type="button" aria-label="关闭" @click="documentsOpen = false">×</button>
        </header>

        <div class="doc-actions">
          <button type="button" class="primary" @click="docUploadInput.click()">
            上传文档
          </button>
          <button type="button" @click="rebuildIndex">重建索引</button>
          <input
            ref="docUploadInput"
            type="file"
            accept=".txt,.md,.pdf,.docx"
            multiple
            hidden
            @change="handleDocUpload"
          />
        </div>
        <p class="doc-hint">支持 TXT / Markdown / PDF / DOCX，单文件不超过 30MB。</p>

        <div v-if="uploadJob" class="job-card">
          <div class="job-label">
            {{ uploadJob.label }} · {{ uploadJob.message }}
          </div>
          <div class="progress-bar">
            <span :style="{ width: `${uploadJob.progress}%` }"></span>
          </div>
        </div>

        <div v-if="notice" class="notice-banner">{{ notice }}</div>
        <div v-if="error" class="error-banner">{{ error }}</div>

        <ul class="doc-list">
          <li v-for="doc in documents" :key="doc.source">
            <div class="doc-info">
              <strong>{{ doc.source }}</strong>
              <small>
                {{ doc.chunks }} 个分块
                <span v-if="doc.first_article">
                  · {{ doc.first_article }} - {{ doc.last_article }}
                </span>
              </small>
            </div>
            <button type="button" class="danger" @click="deleteDocument(doc)">
              删除
            </button>
          </li>
          <li v-if="!documents.length" class="empty-list">暂无文档</li>
        </ul>

        <footer class="doc-drawer-footer">
          重建索引会清空现有向量，并重新入库 data/ 目录下的全部文件。
        </footer>
      </aside>
    </div>
  </div>
</template>
