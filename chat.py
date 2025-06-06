from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

load_dotenv()
client = OpenAI()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_vectors",
    embedding=embedding_model
)

BASE_SYSTEM_PROMPT = """
You are a helpful AI assistant who answers user queries based on the provided context from a PDF file.
Always refer the user to the appropriate page number to find more details. Do not answer anything outside the context.
"""

print("🤖 Ask your questions based on the uploaded PDF. Type 'exit' to quit.\n")

# Persistent message history
messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]

while True:
    try:
        query = input("> ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting. Have a great day!")
            break

        # Vector search for current query
        search_results = vector_db.similarity_search(query=query, k=5)

        # Dynamic context generation
        context = "\n\n".join([
            f"Page Content: {r.page_content}\nPage Number: {r.metadata.get('page_label', 'N/A')}\nSource: {r.metadata.get('source', 'N/A')}"
            for r in search_results
        ])

        # Inject context into system prompt (replaced version for current turn)
        system_prompt_with_context = BASE_SYSTEM_PROMPT + f"\n\nContext:\n{context}"
        
        #override system prompt by prepending again
        turn_messages = [{"role": "system", "content": system_prompt_with_context}] + messages[1:]
        
        # Add user message
        turn_messages.append({"role": "user", "content": query})

        # Get response
        chat_completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=turn_messages
        )

        reply = chat_completion.choices[0].message.content.strip()
        print(f"\n🤖 {reply}\n")

        # Add user and assistant messages to long-term history
        messages.append({"role": "user", "content": query})
        messages.append({"role": "assistant", "content": reply})

    except Exception as e:
        print(f"❌ Error: {e}\n")
