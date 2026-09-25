<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'
import DOMPurify from 'dompurify'
import { marked } from 'marked'

const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const sessions = ref([])
const activeId = ref(null)
const messages = ref([])
const question = ref('')
const busy = ref(false)
const error = ref('')
const fileInput = ref(null)
const chatPane = ref(null)
const activeTitle = computed(() => sessions.value.find(s => s.session_id === activeId.value)?.title || '新会话')

async function loadSessions() {
  const response = await fetch(`${apiBase}/sessions`)
  if (!response.ok) throw new Error('会话列表加载失败')
  sessions.value = await response.json()
}

async function selectSession(session) {
  activeId.value = session.session_id
  messages.value = session.messages || []
  error.value = ''
  await scrollBottom()
}

function newChat() {
  activeId.value = null
  messages.value = []
  error.value = ''
}

async function scrollBottom() {
  await nextTick()
  if (chatPane.value) chatPane.value.scrollTop = chatPane.value.scrollHeight
}

async function send() {
  const text = question.value.trim()
  if (!text || busy.value) return
  error.value = ''
  messages.value.push({ role: 'user', content: text }, { role: 'assistant', content: '' })
  question.value = ''
  busy.value = true
  await scrollBottom()
  try {
    const response = await fetch(`${apiBase}/chat`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: text, session_id: activeId.value }),
    })
    if (!response.ok || !response.body) throw new Error(`请求失败 (${response.status})`)
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let pending = ''
    const answer = messages.value[messages.value.length - 1]
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      pending += decoder.decode(value, { stream: true })
      const blocks = pending.split('\n\n')
      pending = blocks.pop() || ''
      for (const block of blocks) {
        const event = block.match(/^event:\s*(.+)$/m)?.[1]
        const data = block.match(/^data:\s*(.*)$/m)?.[1]
        if (event === 'session' && data) activeId.value = JSON.parse(data).session_id
        else if (event === 'error' && data) throw new Error(JSON.parse(data).detail || '回答生成失败')
        else if (!event && data) answer.content += JSON.parse(data).content || ''
      }
      await scrollBottom()
    }
    await loadSessions()
  } catch (err) {
    error.value = err.message || '服务连接失败'
    const last = messages.value[messages.value.length - 1]
    if (last?.role === 'assistant' && !last.content) messages.value.pop()
  } finally {
    busy.value = false
  }
}

async function uploadFile(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file) return
  error.value = ''
  const form = new FormData()
  form.append('file', file)
  try {
    const response = await fetch(`${apiBase}/upload`, { method: 'POST', body: form })
    const result = await response.json()
    if (!response.ok) throw new Error(result.detail || '文件入库失败')
    error.value = `${result.filename}：${result.result}`
  } catch (err) { error.value = err.message || '文件上传失败' }
}

function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text || ''))
}

onMounted(async () => {
  try {
    await loadSessions()
    if (sessions.value.length) await selectSession(sessions.value[0])
  } catch (err) { error.value = `${err.message}，请确认 FastAPI 已启动` }
})
</script>

<template>
  <div class="layout">
    <aside class="sidebar">
      <div class="brand"><span class="brand-icon">⚖</span><div><strong>法知</strong><small>智能法律问答</small></div></div>
      <button class="new-chat" @click="newChat"><span>＋</span> 新建会话</button>
      <div class="list-title">最近会话</div>
      <nav class="session-list">
        <button v-for="session in sessions" :key="session.session_id" :class="['session', { active: activeId === session.session_id }]" @click="selectSession(session)">
          <span class="session-mark">◷</span><span class="session-name">{{ session.title }}</span>
        </button>
        <div v-if="!sessions.length" class="empty-list">开始一个新问题</div>
      </nav>
      <div class="sidebar-bottom"><span class="status-dot"></span> 本地法律知识库</div>
    </aside>
    <main class="main">
      <header class="topbar"><div class="crumb">法律助手 <span>/</span> {{ activeTitle }}</div><div class="model-badge"><i></i> RAG 已连接</div></header>
      <div class="conversation-shell">
        <section ref="chatPane" class="chat-pane">
          <div v-if="!messages.length" class="welcome">
            <div class="welcome-symbol">⚖</div><h1>你好，有什么法律问题？</h1><p>基于本地法律知识库检索，并在需要时调用外部工具辅助分析。</p>
            <div class="suggestions"><button @click="question='试用期被辞退，用人单位需要支付经济补偿吗？'"><b>劳动关系</b><span>试用期被辞退，是否有经济补偿？</span></button><button @click="question='学校处分学生时，应当遵循哪些程序？'"><b>教育权益</b><span>学校作出处分时应遵循哪些程序？</span></button></div>
          </div>
          <div v-else class="messages">
            <article v-for="(message, index) in messages" :key="index" :class="['message-row', message.role]">
              <div v-if="message.role === 'assistant'" class="avatar assistant-avatar">⚖</div>
              <div :class="['message-content', message.role]">
                <div class="role-label">{{ message.role === 'user' ? '你' : '法律助手' }}</div>
                <div v-if="message.role === 'user'" class="user-text">{{ message.content }}</div>
                <div v-else class="markdown-body" v-html="renderMarkdown(message.content || (busy && index === messages.length - 1 ? '正在分析…' : ''))"></div>
              </div>
              <div v-if="message.role === 'user'" class="avatar user-avatar">你</div>
            </article>
            <div v-if="error" class="error-banner">{{ error }}</div>
          </div>
        </section>
        <footer class="composer-wrap">
          <div v-if="error && !messages.length" class="error-banner standalone">{{ error }}</div>
          <div class="composer">
            <textarea v-model="question" rows="1" placeholder="输入法律问题…" :disabled="busy" @keydown.enter.exact.prevent="send"></textarea>
            <div class="composer-actions"><div class="left-actions"><input ref="fileInput" type="file" accept=".txt,text/plain" hidden @change="uploadFile"><button class="upload-btn" title="上传 TXT 知识文件" @click="fileInput.click()"><span>＋</span> 上传知识文件</button><span class="hint">Enter 发送 · Shift + Enter 换行</span></div><button class="send-btn" :disabled="!question.trim() || busy" @click="send">{{ busy ? '…' : '↑' }}</button></div>
          </div>
          <div class="disclaimer">法律助手提供的信息仅供学习参考，不构成正式法律意见。</div>
        </footer>
      </div>
    </main>
  </div>
</template>
