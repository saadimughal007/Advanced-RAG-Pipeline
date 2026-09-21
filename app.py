import asyncio
import json
from datetime import datetime

import streamlit as st

from rag.advanced_rag import run_advanced_rag


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Advanced RAG Pipeline",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #777;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .answer-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #2ca02c;
        margin: 1rem 0;
    }

    .stage-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e5e5e5;
    }

    .source-box {
        background: #fafafa;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        border: 1px solid #e5e5e5;
        margin: 0.4rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Advanced RAG")

    st.markdown(
        """
        This application demonstrates an optimized RAG pipeline:

        **1. Query Generation**

        **2. Parallel Retrieval**

        **3. Reranking**

        **4. Parallel Compression**

        **5. Final Answer**
        """
    )

    st.divider()

    st.markdown("### Technology Stack")
    st.write("• LangChain")
    st.write("• ChromaDB")
    st.write("• Gemini")
    st.write("• Sentence Transformers")
    st.write("• CrossEncoder")
    st.write("• Streamlit")

    st.divider()

    st.caption(
        "Optimized Advanced RAG Pipeline"
    )


# ============================================================
# HEADER
# ============================================================

col1, col2 = st.columns([4, 1])

with col1:
    st.markdown(
        '<div class="main-title">🚀 Advanced RAG Pipeline</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">'
        "Intelligent Document Retrieval & Answer Generation"
        "</div>",
        unsafe_allow_html=True,
    )

with col2:
    st.caption(
        datetime.now().strftime("%H:%M:%S")
    )


# ============================================================
# INPUT
# ============================================================

st.markdown("### 🔍 Ask Your Question")

question = st.text_area(
    "Question",
    placeholder="Example: How does FastAPI handle high concurrency?",
    height=100,
    label_visibility="collapsed",
)

col1, col2 = st.columns([3, 1])

with col1:
    search_button = st.button(
        "🔍 Search & Generate Answer",
        type="primary",
        use_container_width=True,
    )

with col2:
    clear_button = st.button(
        "🔄 Clear",
        use_container_width=True,
    )

if clear_button:
    st.rerun()


# ============================================================
# RUN PIPELINE
# ============================================================

if search_button:

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Processing")

    try:
        with st.spinner(
            "Running optimized Advanced RAG..."
        ):
            result = asyncio.run(
                run_advanced_rag(
                    question.strip()
                )
            )

        # ====================================================
        # RESULT DATA
        # ====================================================

        answer = result.get("answer", "")
        sources = result.get("sources", [])
        queries = result.get("queries", [])
        metrics = result.get("metrics", {})

        total_time = metrics.get("total", 0)
        query_time = metrics.get(
            "query_generation",
            0,
        )
        retrieval_time = metrics.get(
            "retrieval",
            0,
        )
        rerank_time = metrics.get(
            "reranking",
            0,
        )
        compression_time = metrics.get(
            "compression",
            0,
        )
        final_answer_time = metrics.get(
            "final_answer",
            0,
        )

        # ====================================================
        # PERFORMANCE
        # ====================================================

        st.markdown("### 📊 Performance")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Total",
                f"{total_time:.2f}s",
            )

        with col2:
            st.metric(
                "Query",
                f"{query_time:.2f}s",
            )

        with col3:
            st.metric(
                "Retrieval",
                f"{retrieval_time:.2f}s",
            )

        with col4:
            st.metric(
                "Reranking",
                f"{rerank_time:.2f}s",
            )

        with col5:
            st.metric(
                "Compression",
                f"{compression_time:.2f}s",
            )

        st.caption(
            f"Final answer generation: {final_answer_time:.2f}s"
        )

        # ====================================================
        # PIPELINE STAGES
        # ====================================================

        st.markdown("### 🔬 Pipeline Stages")

        with st.expander(
            "1️⃣ Query Generation",
            expanded=True,
        ):
            if queries:
                for index, query in enumerate(
                    queries,
                    start=1,
                ):
                    st.write(
                        f"**Query {index}:** {query}"
                    )
            else:
                st.info("No queries generated.")

            st.caption(
                f"Stage time: {query_time:.2f}s"
            )

        with st.expander(
            "2️⃣ Parallel Retrieval",
            expanded=False,
        ):
            st.write(
                "Generated queries were searched "
                "in parallel."
            )
            st.caption(
                f"Stage time: {retrieval_time:.2f}s"
            )

        with st.expander(
            "3️⃣ Reranking",
            expanded=False,
        ):
            st.write(
                "Candidate documents were ranked "
                "using the CrossEncoder."
            )
            st.caption(
                f"Stage time: {rerank_time:.2f}s"
            )

        with st.expander(
            "4️⃣ Parallel Compression",
            expanded=False,
        ):
            st.write(
                "Relevant content was extracted "
                "from the top-ranked documents."
            )
            st.caption(
                f"Stage time: {compression_time:.2f}s"
            )

        with st.expander(
            "5️⃣ Final Answer",
            expanded=False,
        ):
            st.write(
                "The final LLM answer was generated "
                "using the compressed context."
            )
            st.caption(
                f"Stage time: {final_answer_time:.2f}s"
            )

        # ====================================================
        # ANSWER
        # ====================================================

        st.markdown("---")
        st.markdown("### ✅ Generated Answer")

        st.markdown(
            f"""
            <div class="answer-box">
                {answer}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ====================================================
        # SOURCES
        # ====================================================

        st.markdown("### 📚 Sources Used")

        if sources:
            for index, source in enumerate(
                sources,
                start=1,
            ):
                filename = (
                    source.split("\\")[-1]
                    if "\\" in source
                    else source.split("/")[-1]
                )

                st.markdown(
                    f"""
                    <div class="source-box">
                        📄 <strong>{index}. {filename}</strong>
                        <br>
                        <small>{source}</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No source documents returned.")

        # ====================================================
        # SUMMARY
        # ====================================================

        st.markdown("### 📋 Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Generated Queries",
                len(queries),
            )

        with col2:
            st.metric(
                "Sources",
                len(sources),
            )

        with col3:
            st.metric(
                "Total Time",
                f"{total_time:.2f}s",
            )

        # ====================================================
        # EXPORT
        # ====================================================

        st.markdown("### 💾 Export")

        export_data = {
            "question": question.strip(),
            "answer": answer,
            "queries": queries,
            "sources": sources,
            "metrics": metrics,
        }

        export_json = json.dumps(
            export_data,
            indent=2,
            default=str,
        )

        export_text = (
            "ADVANCED RAG PIPELINE\n\n"
            f"QUESTION:\n{question.strip()}\n\n"
            f"ANSWER:\n{answer}\n\n"
            "GENERATED QUERIES:\n"
            + "\n".join(
                f"- {query}"
                for query in queries
            )
            + "\n\nSOURCES:\n"
            + "\n".join(
                f"- {source}"
                for source in sources
            )
            + "\n\nMETRICS:\n"
            + "\n".join(
                f"- {key}: {value:.2f}s"
                if isinstance(value, (int, float))
                else f"- {key}: {value}"
                for key, value in metrics.items()
            )
        )

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "📄 Download TXT",
                data=export_text,
                file_name=(
                    f"rag_result_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                ),
                mime="text/plain",
                use_container_width=True,
            )

        with col2:
            st.download_button(
                "📋 Download JSON",
                data=export_json,
                file_name=(
                    f"rag_result_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                ),
                mime="application/json",
                use_container_width=True,
            )

    except Exception as error:
        st.error(
            f"❌ Pipeline Error: {error}"
        )
        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with LangChain, ChromaDB, Gemini, "
    "Sentence Transformers & Streamlit"
)
