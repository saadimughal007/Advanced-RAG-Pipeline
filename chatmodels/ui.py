import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai.chat_models import GoogleRateLimitError

from chatbot import chain


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="AI Tutor",
    page_icon="🤖",
    layout="centered",
)


# ============================================================
# Session State
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []


# ============================================================
# Header
# ============================================================

st.title("🤖 AI Tutor")

st.caption(
    "Ask questions about AI, Python, automation, LangChain, APIs "
    "and software development."
)


# ============================================================
# Display Previous Messages
# ============================================================

for message in st.session_state.display_messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# Stream Response
# ============================================================

def stream_response(user_input: str):

    response_stream = chain.stream({
        "history": st.session_state.chat_history,
        "question": user_input,
    })

    for chunk in response_stream:

        if hasattr(chunk, "content"):

            content = chunk.content

            if isinstance(content, str):
                yield content

            elif isinstance(content, list):

                for part in content:

                    if isinstance(part, str):
                        yield part

                    elif isinstance(part, dict):

                        text = part.get("text")

                        if isinstance(text, str):
                            yield text


# ============================================================
# Chat Input
# ============================================================

if user_input := st.chat_input(
    "Ask a question..."
):

    # --------------------------------------------------------
    # Display User Message
    # --------------------------------------------------------

    st.session_state.display_messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)


    # --------------------------------------------------------
    # Generate Assistant Response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        try:

            full_response = st.write_stream(
                stream_response(user_input)
            )

        except GoogleRateLimitError:

            st.error(
                "Rate limit reached. Please wait and try again."
            )

            full_response = None

        except Exception as e:

            st.error(
                f"Something went wrong: {str(e)}"
            )

            full_response = None


    # --------------------------------------------------------
    # Save Successful Conversation
    # --------------------------------------------------------

    if full_response:

        st.session_state.display_messages.append({
            "role": "assistant",
            "content": full_response,
        })

        st.session_state.chat_history.append(
            HumanMessage(content=user_input)
        )

        st.session_state.chat_history.append(
            AIMessage(content=full_response)
        )
        