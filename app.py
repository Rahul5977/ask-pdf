import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from pathlib import Path
from datetime import datetime
import os
import tempfile


# Load .env variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Ask Your PDF", layout="wide")
st.title("📘 Ask Your PDF")
st.caption("Built with LangChain, Qdrant, OpenAI, Streamlit ")

# Session state for storing messages and collection info
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


# === PDF Upload and Indexing ===
uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])
if uploaded_file:
    with st.spinner("Processing PDF and creating vector embeddings..."):
        # Save temp PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = Path(tmp_file.name)

        # Load + Split
        loader = PyPDFLoader(str(tmp_path))
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
        split_docs = splitter.split_documents(docs)

        # Embedding + Store in Qdrant
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        collection_name = f"collection_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        vector_db = QdrantVectorStore.from_documents(
            documents=split_docs,
            url="http://localhost:6333",
            collection_name=collection_name,
            embedding=embeddings
        )

        st.session_state.vector_db = vector_db
        st.session_state.uploaded_file_name = uploaded_file.name
        st.success("✅ PDF indexed successfully!")




# === Ask Questions ===
if st.session_state.vector_db:
    user_query = st.text_input("💬 Ask a question based on the PDF:")

    if user_query:
        with st.spinner("Thinking..."):
            try:
                search_results = st.session_state.vector_db.similarity_search(query=user_query, k=5)
                context = "\n\n".join([
                    f"Page Content: {r.page_content}\nPage Number: {r.metadata.get('page_label', 'N/A')}"
                    for r in search_results
                ])

                SYSTEM_PROMPT = f"""
                You are a helpful AI assistant who answers user queries based on the provided context from a PDF file.
                Always refer the user to the correct page number. Don't hallucinate.
                
                Context:
                {context}
                """

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query}
                ]

                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=messages
                )
                reply = response.choices[0].message.content.strip()

                # Save
                st.session_state.messages.append({"role": "user", "content": user_query})
                st.session_state.messages.append({"role": "assistant", "content": reply})

                st.markdown(f"**🤖 Assistant:** {reply}")
                
                # Context toggle
                with st.expander("📖 Show GPT Context (top 5 chunks)"):
                    st.code(context)


            except Exception as e:
                st.error(f"Error: {e}")

    # Display full history
    with st.expander("🕘 Chat History"):
        for i in range(0, len(st.session_state.messages), 2):
            st.markdown(f"**🧑 You:** {st.session_state.messages[i]['content']}")
            st.markdown(f"**🤖 Assistant:** {st.session_state.messages[i+1]['content']}")
    
    # Download chat history
    if st.download_button("💾 Download Chat History", 
        data="\n\n".join([f"You: {m['content']}" if i % 2 == 0 else f"Assistant: {m['content']}" 
                          for i, m in enumerate(st.session_state.messages)]),
        file_name="chat_history.txt"):
        st.success("Downloaded chat history!")

else:
    st.info("⬆️ Upload a PDF first to start asking questions.")


st.caption("Made with ❤️ by Rahul")