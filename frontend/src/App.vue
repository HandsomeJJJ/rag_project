<script setup>
import { computed, nextTick, onMounted, ref } from 'vue';
import DOMPurify from 'dompurify';
import { marked } from 'marked';

const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const sessions = ref([]);
const activeId = ref(null);
const messages = ref([]);
const question = ref('');
const busy = ref(false);
const error = ref('');
const fileInput = ref(null);
const chatPane = ref(null);
const searchText = ref('');
const editingSessionId = ref(null);
const editingTitle = ref('');
const openMenuSessionId = ref(null);
const copiedMessageIndex = ref(null);
const activeController = ref(null);
const activeTitle = computed(
  () =>
    sessions.value.find((s) => s.session_id === activeId.value)?.title ||
    '新会话'
);
const visibleSessions = computed(() => {
  const query = searchText.value.trim().toLocaleLowerCase();
  if (!query) return sessions.value;
  return sessions.value.filter((session) =>
    session.title.toLocaleLowerCase().includes(query)
  );
});

async function loadSessions() {
  const response = await fetch(`${apiBase}/sessions`);
  if (!response.ok) throw new Error('会话列表加载失败');
  sessions.value = await response.json();
}

async function selectSession(session) {
  if (editingSessionId.value) return;
  if (busy.value) stopGeneration();
  activeId.value = session.session_id;
  messages.value = session.messages || [];
  error.value = '';
  openMenuSessionId.value = null;
  await scrollBottom();
}

function newChat() {
  if (busy.value) stopGeneration();
  activeId.value = null;
  messages.value = [];
  error.value = '';
  openMenuSessionId.value = null;
  editingSessionId.value = null;
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
    const response = await fetch(
      `${apiBase}/sessions/${encodeURIComponent(session.session_id)}`,
      {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title }),
      }
    );
    const result = await response.json();
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
    const response = await fetch(
      `${apiBase}/sessions/${encodeURIComponent(session.session_id)}/pin`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pinned: !session.pinned }),
      }
    );
    const result = await response.json();
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
    const response = await fetch(
      `${apiBase}/sessions/${encodeURIComponent(session.session_id)}`,
      { method: 'DELETE' }
    );
    const result = response.ok ? null : await response.json();
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
      { role: 'assistant', content: '' }
    );
    answer = messages.value[messages.value.length - 1];
  } else {
    answer = messages.value[replaceIndex];
    if (!answer || answer.role !== 'assistant') return;
    previousAnswer = answer.content;
    answer.content = '';
  }
  busy.value = true;
  const controller = new AbortController();
  activeController.value = controller;
  await scrollBottom();
  try {
    const response = await fetch(`${apiBase}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        question: text,
        session_id: activeId.value,
        ...(replaceIndex === null
          ? {}
          : { regenerate_message_index: replaceIndex }),
      }),
    });
    if (!response.ok || !response.body)
      throw new Error(`请求失败 (${response.status})`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      const blocks = pending.split('\n\n');
      pending = blocks.pop() || '';
      for (const block of blocks) {
        const event = block.match(/^event:\s*(.+)$/m)?.[1];
        const data = block.match(/^data:\s*(.*)$/m)?.[1];
        if (event === 'session' && data && !controller.signal.aborted)
          activeId.value = JSON.parse(data).session_id;
        else if (event === 'error' && data)
          throw new Error(JSON.parse(data).detail || '回答生成失败');
        else if (!event && data)
          answer.content += JSON.parse(data).content || '';
      }
      await scrollBottom();
    }
    await loadSessions();
    if (activeId.value) {
      const refreshed = sessions.value.find(
        (session) => session.session_id === activeId.value
      );
      if (refreshed) messages.value = refreshed.messages;
    }
  } catch (err) {
    if (err.name === 'AbortError') return;
    error.value = err.message || '服务连接失败';
    if (replaceIndex !== null) answer.content = previousAnswer;
    else if (!answer.content) messages.value.pop();
  } finally {
    busy.value = false;
    activeController.value = null;
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

async function uploadFile(event) {
  const file = event.target.files?.[0];
  event.target.value = '';
  if (!file) return;
  error.value = '';
  const form = new FormData();
  form.append('file', file);
  try {
    const response = await fetch(`${apiBase}/upload`, {
      method: 'POST',
      body: form,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || '文件入库失败');
    error.value = `${result.filename}：${result.result}`;
  } catch (err) {
    error.value = err.message || '文件上传失败';
  }
}

function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text || ''));
}

onMounted(async () => {
  try {
    await loadSessions();
    if (sessions.value.length) await selectSession(sessions.value[0]);
  } catch (err) {
    error.value = `${err.message}，请确认 FastAPI 已启动`;
  }
});
</script>

<template>
  <!-- 页面最外层容器 -->
  <div class="layout">
    <!-- ========== 左侧边栏 ========== -->
    <aside class="sidebar">
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
        <span class="status-dot"></span> 本地法律知识库
      </div>
    </aside>

    <!-- ========== 右侧主区域 ========== -->
    <main class="main">
      <!-- 顶部导航栏 -->
      <header class="topbar">
        <div class="crumb">法律助手 <span>/</span> {{ activeTitle }}</div>
        <div class="model-badge"><i></i> RAG 已连接</div>
      </header>

      <div class="conversation-shell">
        <!-- 聊天面板容器，ref用于js拿到DOM，自动滚动到底部 -->
        <section ref="chatPane" class="chat-pane">
          <!-- 没有消息的时候：欢迎首页 + 推荐问题按钮 -->
          <div v-if="!messages.length" class="welcome">
            <div class="welcome-symbol">⚖</div>
            <h1>你好，有什么法律问题？</h1>
            <p>基于本地法律知识库检索，并在需要时调用外部工具辅助分析。</p>
            <div class="suggestions">
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
                  v-html="
                    renderMarkdown(
                      message.content ||
                        (busy && index === messages.length - 1
                          ? '正在分析…'
                          : '')
                    )
                  "
                ></div>
                <div
                  v-if="message.role === 'assistant' && message.content && !busy"
                  class="message-actions"
                >
                  <button type="button" @click="copyAnswer(index, message.content)">
                    {{ copiedMessageIndex === index ? '已复制' : '复制' }}
                  </button>
                  <button type="button" @click="regenerate(index)">重新生成</button>
                </div>
              </div>
              <!-- 用户消息：右侧头像 -->
              <div v-if="message.role === 'user'" class="avatar user-avatar">
                🙂
              </div>
            </article>
            <!-- 全局错误提示条 -->
            <div v-if="error" class="error-banner">{{ error }}</div>
          </div>
        </section>

        <!-- 底部输入区域 -->
        <footer class="composer-wrap">
          <!-- 空会话时单独展示错误 -->
          <div v-if="error && !messages.length" class="error-banner standalone">
            {{ error }}
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
                  accept=".txt,text/plain"
                  hidden
                  @change="uploadFile"
                />
                <button
                  class="upload-btn"
                  title="上传 TXT 知识文件"
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
  </div>
</template>
