import os
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv

from mlb_multiturn_agent import build_agent


APP_TITLE = "MLB 멀티턴 에이전트 데모 (Streamlit)"


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # [{"role":"user|assistant", "content":"..."}]
    if "agent" not in st.session_state:
        st.session_state["agent"] = None


def ensure_agent() -> None:
    # .env 로드 후 환경변수에서 키 사용
    load_dotenv(override=False)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        st.error("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 또는 Windows 환경변수를 설정해 주세요.")
        return
    # LangChain이 환경변수에서 읽도록 보장
    os.environ["OPENAI_API_KEY"] = api_key
    if st.session_state.get("agent") is None:
        try:
            st.session_state["agent"] = build_agent()
        except Exception as e:
            st.error(f"에이전트 초기화 실패: {e}")


def reset_conversation() -> None:
    st.session_state["messages"] = []
    if st.session_state.get("agent") is not None:
        try:
            st.session_state["agent"].memory.clear()
        except Exception:
            st.session_state["agent"] = None


def render_history(messages: List[Dict[str, str]]) -> None:
    for m in messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="⚾")
    st.title(APP_TITLE)
    st.caption("LangChain 에이전트 + MLB-StatsAPI 도구를 이용한 간단 챗 UI")

    init_state()

    # 상단 조작 (사이드바 없이)
    top_left, top_right = st.columns([1, 1])
    with top_left:
        if st.button("대화 초기화"):
            reset_conversation()
            st.experimental_rerun()
    with top_right:
        model_env = os.getenv("OPENAI_MODEL")
        shown_model = model_env if model_env else "gpt-4o-mini (default)"
        st.write(f"모델: {shown_model}")

    ensure_agent()

    # 기존 대화 이력 표시
    render_history(st.session_state["messages"])

    # 입력창
    prompt = st.chat_input("질문을 입력하세요 (예: 오타니 2023시즌 타격 기록 보여줘)")
    if not prompt:
        return

    # 사용자 메시지 반영
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 에이전트 응답
    with st.chat_message("assistant"):
        if st.session_state.get("agent") is None:
            st.error("에이전트가 준비되지 않았습니다. 환경변수 설정을 확인해 주세요.")
            return
        try:
            result = st.session_state["agent"].invoke({"input": prompt})
            output = (result or {}).get("output", "")
        except Exception as e:
            err_msg = str(e)
            lowered = err_msg.lower()
            needs_fallback = any(
                k in lowered for k in [
                    "invalid model",
                    "does not exist",
                    "model not found",
                    "unsupported",
                    "unknown model",
                ]
            )
            if needs_fallback:
                fallback_model = "gpt-4o-mini"
                try:
                    os.environ["OPENAI_MODEL"] = fallback_model
                    st.session_state["agent"] = build_agent()
                    st.info(f"모델 문제가 발생하여 '{fallback_model}'으로 대체했습니다.")
                    result = st.session_state["agent"].invoke({"input": prompt})
                    output = (result or {}).get("output", "")
                except Exception as e2:
                    output = f"오류가 발생했습니다: {e2}"
            else:
                output = f"오류가 발생했습니다: {e}"
        st.write(output)
    st.session_state["messages"].append({"role": "assistant", "content": output})


if __name__ == "__main__":
    main()
