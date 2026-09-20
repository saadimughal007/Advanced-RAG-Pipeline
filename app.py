import streamlit as st
from rag.complete_advanced_rag import run_advanced_rag
import time
from datetime import datetime

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
    /* Main theme */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --danger-color: #d62728;
    }
    
    /* Header styling */
    .header-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .metric-card-green {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .metric-card-blue {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    
    .metric-number {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Pipeline stage styling */
    .pipeline-stage {
        background: white;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .stage-title {
        font-weight: bold;
        font-size: 1.1rem;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    
    .stage-content {
        color: #333;
        font-size: 0.95rem;
    }
    
    /* Answer box */
    .answer-box {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #2ca02c;
        margin: 2rem 0;
    }
    
    .answer-text {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #333;
    }
    
    /* Source cards */
    .source-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .source-name {
        font-weight: bold;
        color: #1f77b4;
        font-size: 0.95rem;
    }
    
    /* Divider */
    .divider {
        border-top: 2px solid #e0e0e0;
        margin: 2rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("---")
    
    st.markdown("#### About This App")
    st.info(
        """
        **Advanced RAG Pipeline** combines:
        - 🔄 Query Rewriting
        - 📚 Multi-Query Retrieval
        - 🎯 Reranking (CrossEncoder)
        - ✂️ Context Compression
        - 🧠 LLM-Powered Answers
        
        Perfect for Q&A over documents!
        """
    )
    
    st.markdown("---")
    
    st.markdown("#### Technology Stack")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Backend:**")
        st.write("• LangChain")
        st.write("• ChromaDB")
        st.write("• Gemini API")
    with col2:
        st.write("**Frontend:**")
        st.write("• Streamlit")
        st.write("• Python 3.11+")
        st.write("• CrossEncoder")
    
    st.markdown("---")
    
    st.markdown("#### Links")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.link_button("GitHub", "https://github.com", use_container_width=True)
    with col2:
        st.link_button("LinkedIn", "https://linkedin.com", use_container_width=True)
    with col3:
        st.link_button("Portfolio", "https://portfolio.com", use_container_width=True)

# ============================================================
# MAIN CONTENT
# ============================================================

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(
        '<div class="header-title">🚀 Advanced RAG Pipeline</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="header-subtitle">Intelligent Document Retrieval & Answer Generation</div>',
        unsafe_allow_html=True,
    )

with col2:
    st.markdown("### ")
    st.markdown("### ")
    current_time = datetime.now().strftime("%H:%M:%S")
    st.text(f"🕐 {current_time}")

st.markdown("---")

# ============================================================
# INPUT SECTION
# ============================================================

st.markdown("### 🔍 Ask Your Question")

question = st.text_area(
    "Enter your question:",
    placeholder="Example: How can I make my Python API handle lots of users?",
    height=100,
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    search_button = st.button(
        "🔍 Search & Generate Answer",
        use_container_width=True,
        type="primary",
    )

with col2:
    clear_button = st.button("🔄 Clear", use_container_width=True)

with col3:
    st.write("")

if clear_button:
    st.rerun()

# ============================================================
# PROCESSING
# ============================================================

if search_button and question.strip():
    
    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Processing")
    
    # Create placeholders for progress
    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()
    pipeline_placeholder = st.empty()
    result_placeholder = st.empty()
    
    # Start timer
    start_time = time.time()
    
    # Run pipeline
    try:
        with st.spinner("🔄 Running Advanced RAG Pipeline..."):
            result = run_advanced_rag(question)
        
        end_time = time.time()
        execution_time = round(end_time - start_time, 2)
        
        # ============================================================
        # METRICS SECTION
        # ============================================================
        
        with metrics_placeholder.container():
            st.markdown("### 📊 Pipeline Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Execution Time</div>
                        <div class="metric-number">{execution_time}s</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col2:
                total_retrieved = result["pipeline_info"]["total_retrieved"]
                st.markdown(
                    f"""
                    <div class="metric-card metric-card-orange">
                        <div class="metric-label">Documents Retrieved</div>
                        <div class="metric-number">{total_retrieved}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col3:
                reranked = result["pipeline_info"]["reranked_top_k"]
                st.markdown(
                    f"""
                    <div class="metric-card metric-card-green">
                        <div class="metric-label">After Reranking</div>
                        <div class="metric-number">{reranked}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col4:
                compressed = result["pipeline_info"]["compressed_documents"]
                st.markdown(
                    f"""
                    <div class="metric-card metric-card-blue">
                        <div class="metric-label">Compressed Docs</div>
                        <div class="metric-number">{compressed}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        st.markdown("---")
        
        # ============================================================
        # PIPELINE STAGES SECTION
        # ============================================================
        
        with st.expander("📋 Pipeline Stages (Detailed)", expanded=False):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Original Question")
                st.markdown(
                    f"""
                    <div class="pipeline-stage">
                        <div class="stage-content">{question}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                st.markdown("#### Stage 1️⃣: Query Rewriting")
                rewritten = result["pipeline_info"]["rewritten_query"]
                st.markdown(
                    f"""
                    <div class="pipeline-stage">
                        <div class="stage-content"><strong>Optimized:</strong><br>{rewritten}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col2:
                st.markdown("#### Stage 2️⃣: Multi-Query Generation")
                queries = result["pipeline_info"]["generated_queries"]
                queries_str = "<br>".join([f"• {q}" for q in queries])
                st.markdown(
                    f"""
                    <div class="pipeline-stage">
                        <div class="stage-content"><strong>Generated Queries:</strong><br>{queries_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                st.markdown("#### Stage 3️⃣: Reranking Summary")
                st.markdown(
                    f"""
                    <div class="pipeline-stage">
                        <div class="stage-content">
                            <strong>Retrieval Summary:</strong><br>
                            • Total Retrieved: {result['pipeline_info']['total_retrieved']}<br>
                            • After Reranking: {result['pipeline_info']['reranked_top_k']}<br>
                            • After Compression: {result['pipeline_info']['compressed_documents']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        st.markdown("---")
        
        # ============================================================
        # ANSWER SECTION
        # ============================================================
        
        st.markdown("### ✅ Generated Answer")
        
        st.markdown(
            f"""
            <div class="answer-box">
                <div class="answer-text">{result['answer']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("---")
        
        # ============================================================
        # SOURCES SECTION
        # ============================================================
        
        st.markdown("### 📚 Sources Used")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            for i, source in enumerate(result["sources"], 1):
                # Extract filename from full path
                filename = source.split("\\")[-1] if "\\" in source else source.split("/")[-1]
                
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div style="display: flex; align-items: center;">
                            <span style="font-size: 1.5rem; margin-right: 0.5rem;">📄</span>
                            <div>
                                <div class="source-name">{i}. {filename}</div>
                                <div style="font-size: 0.8rem; color: #999;">{source}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        with col2:
            st.markdown("#### 📊 Summary")
            st.metric("Total Sources", len(result["sources"]))
            st.metric("Processing Time", f"{execution_time}s")
            st.metric("Documents Analyzed", result["pipeline_info"]["total_retrieved"])
        
        st.markdown("---")
        
        # ============================================================
        # EXPORT SECTION
        # ============================================================
        
        st.markdown("### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare text export
            export_text = f"""
ADVANCED RAG PIPELINE - RESULTS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

QUESTION:
{question}

ANSWER:
{result['answer']}

SOURCES:
{chr(10).join([f'- {s}' for s in result['sources']])}

PIPELINE METRICS:
- Execution Time: {execution_time}s
- Documents Retrieved: {result['pipeline_info']['total_retrieved']}
- After Reranking: {result['pipeline_info']['reranked_top_k']}
- Compressed Documents: {result['pipeline_info']['compressed_documents']}
"""
            
            st.download_button(
                label="📄 Download as Text",
                data=export_text,
                file_name=f"rag_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        
        with col2:
            # Prepare JSON export
            import json
            export_json = json.dumps(result, indent=2)
            
            st.download_button(
                label="📋 Download as JSON",
                data=export_json,
                file_name=f"rag_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Kripya apna sawal dobara likho ya baad mein try karo.")

elif search_button and not question.strip():
    st.warning("⚠️ Kripya ek sawal likho!")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #999; font-size: 0.9rem;'>
        Built with ❤️ using LangChain, ChromaDB, and Streamlit<br>
        Advanced RAG Pipeline v1.0 | 2024
    </div>
    """,
    unsafe_allow_html=True,
)
