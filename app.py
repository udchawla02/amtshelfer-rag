"""Streamlit UI for AmtsHelfer.

Run:  streamlit run app.py

Two modes:
  * Ask        -> grounded Q&A over your documents (hybrid retrieval + rerank)
  * Triage     -> paste a letter, get a red/yellow/green urgency verdict

Keep this thin: all real logic lives in src/.
"""

from __future__ import annotations

import streamlit as st

from config import settings
from src.rag import RagChain
from src.triage import triage_letter

st.set_page_config(page_title="AmtsHelfer", page_icon="📄", layout="centered")

LANGUAGES = ["English", "German", "Arabic", "Turkish", "Hindi", "Ukrainian", "French", "Spanish"]


@st.cache_resource
def get_chain() -> RagChain:
    return RagChain()


st.title("📄 AmtsHelfer")
st.caption(
    "Understand German admin & health-insurance documents. "
    f"Backend: **{settings.backend}** · hybrid retrieval + reranking · answers grounded in your docs."
)

with st.sidebar:
    st.header("Settings")
    language = st.selectbox("Answer language", LANGUAGES, index=0)
    st.caption(
        f"Hybrid retrieval: {'on' if settings.use_hybrid else 'off'}  ·  "
        f"Reranker: {'on' if settings.use_reranker else 'off'}"
    )

tab_ask, tab_triage = st.tabs(["💬 Ask", "🚦 Triage a letter"])

# ----------------------------- Ask tab ---------------------------------------
with tab_ask:
    try:
        chain = get_chain()
    except Exception as exc:  # noqa: BLE001
        st.error(
            "Could not load the index. Run `python -m src.ingest` first.\n\n"
            f"Details: {exc}"
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("e.g. How long do I have to register my address?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                ans = chain.ask(prompt, language=language)
            st.markdown(ans.text)
            if ans.sources:
                st.caption("Retrieved from: " + ", ".join(ans.sources))
        st.session_state.messages.append({"role": "assistant", "content": ans.text})

# --------------------------- Triage tab --------------------------------------
with tab_triage:
    st.write("Paste the text of a letter you received and get an urgency verdict.")
    letter = st.text_area("Letter text", height=200, placeholder="Sehr geehrte Damen und Herren, ...")
    if st.button("Triage", type="primary") and letter.strip():
        with st.spinner("Reading the letter..."):
            t = triage_letter(letter, language=language)
        colour = {"red": "red", "yellow": "orange", "green": "green"}.get(t.level, "gray")
        st.markdown(f"### {t.emoji} :{colour}[{t.level.upper()}] — {t.title}")
        if t.deadline:
            st.warning(f"**Deadline:** {t.deadline}")
        st.markdown(f"**Why:** {t.reason}")
        st.info(f"**Do this next:** {t.action}")
        st.caption("Not legal advice — always confirm with the relevant office.")
