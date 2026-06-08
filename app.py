import streamlit as st
import os
import time
import groq
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. НАСТРОЙКА СТРАНИЦЫ И ИНТЕРФЕЙСА
# ==========================================
st.set_page_config(page_title="Multi-Agent Content Factory", page_icon="🤖", layout="wide")
st.title("🤖 Мультиагентная фабрика контента")
st.subheader("Автономная универсальная система генерации статей на базе LangGraph")

# Проверяем наличие ключей в системе (для Streamlit Cloud)
if not os.environ.get("GROQ_API_KEY") or not os.environ.get("TAVILY_API_KEY"):
    st.error("❌ Ключи API не найдены. Проверьте настройки Secrets!")
    st.stop()

# ==========================================
# 2. ОПТИМИЗИРОВАННАЯ ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
# ==========================================
@st.cache_resource
def init_llms():
    # Сильный Автор для красивого слога (Llama 3.3 70B)
    writer_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    # Супер-экономный Редактор для экономии суточных лимитов токенов (Llama 3.1 8B Instant)
    editor_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
    return writer_llm, editor_llm

writer_llm, editor_llm = init_llms()

# ==========================================
# 3. ОПРЕДЕЛЕНИЕ ОБЩЕГО СОСТОЯНИЯ (STATE)
# ==========================================
class AgentState(TypedDict):
    topic: str
    research_data: str
    draft: str
    review_comments: str
    revision_count: int
    error_message: str  # Поле для проброса ошибок лимитов наружу

# ==========================================
# 4. УНИВЕРСАЛЬНЫЕ УЗЛЫ АГЕНТОВ С ОБРАБОТКОЙ ЛИМИТОВ
# ==========================================

# --- Узел 1: Исследователь ---
def researcher_node(state: AgentState):
    st.info(f"🔍 **[Исследователь]:** Запускаю поиск в интернете по теме: '{state['topic']}'...")
    try:
        search_tool = TavilySearchResults(max_results=4)
        search_results = search_tool.invoke({"query": state["topic"]})
        
        formatted_results = ""
        for idx, res in enumerate(search_results, 1):
            formatted_results += f"Источник {idx} ({res['url']}):\n{res['content']}\n\n"
        return {"research_data": formatted_results, "error_message": ""}
    except Exception as e:
        st.error(f"Ошибка поиска Tavily: {e}")
        return {"research_data": f"Не удалось собрать данные из-за ошибки: {e}", "error_message": ""}

# --- Узел 2: Писатель ---
def writer_node(state: AgentState):
    if state.get("error_message"):  # Если на прошлом шаге была критическая ошибка — пропускаем
        return {}
        
    st.info("✍️ **[Писатель]:** Создаю/корректирую черновик статьи...")
    from langchain_core.prompts import ChatPromptTemplate
    
    writer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты — профессиональный универсальный автор статей, копирайтер и глубокий эксперт.\n"
         "Твоя задача — написать качественную, структурированную статью в формате Markdown.\n\n"
         "ПРАВИЛА:\n"
         "- Адаптируй тон повествования под тему (экспертный для IT, живой для лайфстайла, аналитический для бизнеса).\n"
         "- Используй четкие заголовки (##, ###) и списки.\n"
         "- Максимально опирайся на факты и цифры из предоставленных данных исследования.\n"
         "- Структура: Введение, Основная часть, Заключение.\n"
         "- Язык: русский."),
        ("user",
         "Тема статьи: {topic}\n\n"
         "Данные из сети для интеграции:\n{research_data}\n\n"
         "Замечания редактора (если есть):\n{review_comments}\n\n"
         "Напиши полную версию статьи:")
    ])
    
    chain = writer_prompt | writer_llm
    try:
        response = chain.invoke({
            "topic": state["topic"],
            "research_data": state["research_data"],
            "review_comments": state.get("review_comments", "Замечаний пока нет.")
        })
        return {"draft": response.content, "error_message": ""}
    except Exception as e:
        # Проверяем, является ли ошибка превышением лимитов Groq
        if "rate_limit_exceeded" in str(e).lower() or isinstance(e, groq.RateLimitError):
            return {"error_message": f"RATE_LIMIT_WRITER::{str(e)}"}
        raise e

# --- Узел 3: Редактор ---
def editor_node(state: AgentState):
    if state.get("error_message"):
        return {}
        
    st.info("🧐 **[Редактор]:** Оцениваю качество текста...")
    from langchain_core.prompts import ChatPromptTemplate
    
    editor_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты — главный редактор контент-платформы. Оцени статью по теме '{topic}' по критериям:\n"
         "1. Соответствие заголовку | 2. Наличие фактов из поиска | 3. Логика, введение и выводы.\n\n"
         "ПРАВИЛО ОТВЕТА:\n"
         "- Если статья отличная, верни ТОЛЬКО ОДНО слово 'ОДОБРЕНО'.\n"
         "- Если текст нужно улучшить, напиши краткие замечания списком (не более 3 предложений суммарно). Без лишних слов!"),
        ("user", "Оцени черновик:\n\n{draft}")
    ])
    
    chain = editor_prompt | editor_llm
    try:
        response = chain.invoke({
            "topic": state["topic"],
            "draft": state["draft"]
        })
        result = response.content.strip()
        
        current_revisions = state.get("revision_count", 0) + 1
        
        if "ОДОБРЕНО" in result.upper() or current_revisions >= 2:
            if current_revisions >= 2 and "ОДОБРЕНО" not in result.upper():
                st.warning("🛑 **[Система]:** Достигнут лимит итераций (2/2) для защиты лимитов токенов. Выводим финал.")
            else:
                st.success("🎉 **[Редактор]:** Статья великолепна и одобрена!")
            return {"review_comments": "", "revision_count": current_revisions, "error_message": ""}
        
        st.warning(f"⚠️ **[Редактор] вернул на доработку:** {result}")
        return {"review_comments": result, "revision_count": current_revisions, "error_message": ""}
    except Exception as e:
        if "rate_limit_exceeded" in str(e).lower() or isinstance(e, groq.RateLimitError):
            return {"error_message": f"RATE_LIMIT_EDITOR::{str(e)}"}
        raise e

# --- Функция ветвления ---
def should_continue(state: AgentState) -> str:
    if state.get("error_message"):
        return "end"
    if state.get("review_comments"):
        return "writer"
    return "end"

# ==========================================
# 5. СБОРКА И КОМПИЛЯЦИЯ МУЛЬТИАГЕНТНОГО ГРАФА
# ==========================================
@st.cache_resource
def compile_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("editor", editor_node)
    
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "editor")
    workflow.add_conditional_edges("editor", should_continue, {"writer": "writer", "end": END})
    
    return workflow.compile(checkpointer=MemorySaver())

app = compile_graph()

# ==========================================
# 6. ЭЛЕМЕНТЫ ИНТЕРФЕЙСА (UI STREAMLIT)
# ==========================================
topic_input = st.text_input("Введите любую тему для статьи:", placeholder="Например: Тренды бэкенда на Go или Как вырастить суккуленты")

if st.button("Запустить фабрику агентов 🚀", type="primary"):
    if not topic_input.strip():
        st.warning("Пожалуйста, введите тему!")
    else:
        st.write("---")
        st.subheader("⚙️ Лог работы агентов в реальном времени:")
        
        log_container = st.container()
        
        with log_container:
            inputs = {"topic": topic_input, "error_message": ""}
            config = {"configurable": {"thread_id": "streamlit_thread"}}
            
            # Запуск стриминга графа
            for output in app.stream(inputs, config):
                pass
            
            final_state = app.get_state(config)
            err_msg = final_state.values.get("error_message", "")
            
            # --- ОБРАБОТКА КРАСИВОГО ТАЙМЕРА ЕСЛИ ЕСТЬ ОШИБКА ЛИМИТОВ ---
            if err_msg.startswith("RATE_LIMIT"):
                st.write("---")
                # Пытаемся вытащить время ожидания из текста ошибки (Groq обычно пишет "try again in XmYs" или "Please try again in X.XXs.")
                import re
                
                # Ищем структуру минут и секунд (например, 1m23s или просто секунды 45s)
                time_find = re.findall(r'try again in (\d+\.?\d*m?s?)', err_msg)
                
                wait_seconds = 60  # Значение по умолчанию, если не распарсили текст
                
                if time_find:
                    raw_time = time_find[0]
                    if 'm' in raw_time:
                        parts = raw_time.split('m')
                        minutes = int(parts[0])
                        seconds = int(parts[1].replace('s', '')) if parts[1] else 0
                        wait_seconds = (minutes * 60) + seconds
                    else:
                        wait_seconds = int(float(raw_time.replace('s', '')))
                
                st.error("⏳ **Достигнут суточный лимит запросов к бесплатному API Groq!**")
                
                # Создаем красивый динамический прогресс-бар ожидания
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                total_wait = wait_seconds
                for i in range(total_wait, 0, -1):
                    mins, secs = divmod(i, 60)
                    status_text.warning(f"🔄 Пожалуйста, подождите. Лимиты обновятся через: **{mins:02d}:{secs:02d}**")
                    # Заполняем прогресс бар
                    progress_bar.progress((total_wait - i) / total_wait)
                    time.sleep(1)
                
                status_text.success("✅ Время ожидания истекло! Вы можете запустить генерацию снова.")
                progress_bar.empty()
                
            else:
                # Если всё прошло успешно, выводим статью
                final_draft = final_state.values.get("draft", "Ошибка генерации.")
                st.write("---")
                st.subheader("📄 Финальный результат статьи:")
                st.markdown(final_draft)
