# ============================================================
# CampusGPT - PDF RAG Chatbot
# Local Embeddings + FAISS + Gemini Generation
#
# Architecture:
# PDF
#   ↓
# Extract text
#   ↓
# Split into chunks
#   ↓
# Local Sentence Transformer embeddings
#   ↓
# FAISS similarity search
#   ↓
# Retrieve relevant chunks
#   ↓
# Gemini generates final answer
#
# No campus_faiss.index
# No chunks.pkl
# No Gemini embedding API
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import numpy as np
import faiss

from dotenv import load_dotenv
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer

from google import genai
from google.genai import types


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set. "
        "Please check your .env file."
    )


# ============================================================
# 3. GEMINI CONFIGURATION
# ============================================================

client = genai.Client(
    api_key=GOOGLE_API_KEY
)

GEN_MODEL_NAME = "gemini-3.8-flash"

print("Gemini API configured successfully.")


# ============================================================
# 4. LOCAL EMBEDDING MODEL
# ============================================================

print("\nLoading local embedding model...")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(
    EMBED_MODEL_NAME
)

print("Local embedding model loaded successfully.")


# ============================================================
# 5. PDF LOADER
# ============================================================

def load_pdf(file_path):
    """
    Extract text from a PDF page by page.

    Returns:
        [
            {
                "source": "sample.pdf",
                "page": 1,
                "text": "..."
            },
            ...
        ]
    """

    reader = PdfReader(file_path)

    source_name = os.path.basename(file_path)

    pages_data = []

    for i, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        text = text.strip()

        if text:

            pages_data.append(
                {
                    "source": source_name,
                    "page": i,
                    "text": text
                }
            )

    print(
        f"Loaded '{source_name}' -> "
        f"{len(pages_data)} pages with text."
    )

    return pages_data


# ============================================================
# 6. LOAD MULTIPLE PDFs
# ============================================================

def load_multiple_pdfs(file_paths):
    """
    Loads multiple PDF files and combines
    all pages into one list.
    """

    all_pages_data = []

    for path in file_paths:

        all_pages_data.extend(
            load_pdf(path)
        )

    return all_pages_data


# ============================================================
# 7. TEXT CHUNKING
# ============================================================

def chunk_text(
    pages_data,
    chunk_size=150,
    overlap=30
):
    """
    Splits each PDF page into overlapping
    word-based chunks.

    chunk_size:
        Number of words in each chunk.

    overlap:
        Number of words shared between
        consecutive chunks.
    """

    chunks = []

    chunk_id = 0

    for page_data in pages_data:

        source = page_data["source"]

        page = page_data["page"]

        text = page_data["text"]

        words = text.split()

        start = 0

        while start < len(words):

            end = min(
                start + chunk_size,
                len(words)
            )

            chunk = " ".join(
                words[start:end]
            )

            chunks.append(
                {
                    "id": chunk_id,
                    "source": source,
                    "page": page,
                    "text": chunk
                }
            )

            chunk_id += 1

            # Stop at final chunk
            if end >= len(words):
                break

            # Keep overlap
            start = end - overlap

    print(
        f"Created {len(chunks)} chunks."
    )

    return chunks


# ============================================================
# 8. LOCAL EMBEDDINGS
# ============================================================

def embed_texts(texts):
    """
    Creates embeddings using the local
    Sentence Transformer model.

    IMPORTANT:
    This does NOT call Gemini.
    """

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


# ============================================================
# 9. BUILD FAISS INDEX
# ============================================================

def build_vector_index(chunks):
    """
    Creates FAISS index from chunks.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        "\nCreating local embeddings..."
    )

    embeddings = embed_texts(
        texts
    )

    dimension = embeddings.shape[1]

    print(
        f"Embedding dimension: "
        f"{dimension}"
    )

    # Inner Product works like cosine similarity
    # because embeddings are normalized.
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    print(
        f"FAISS index built with "
        f"{index.ntotal} vectors."
    )

    return index


# ============================================================
# 10. RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_relevant_chunks(
    question,
    index,
    chunks,
    top_k=4
):
    """
    Finds the most relevant PDF chunks
    for the user's question.
    """

    # Create embedding for user question
    query_embedding = embed_texts(
        [question]
    )

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        chunk = chunks[idx].copy()

        chunk["score"] = float(
            score
        )

        results.append(
            chunk
        )

    return results


# ============================================================
# 11. BUILD PROMPT
# ============================================================

def build_prompt(
    question,
    relevant_chunks,
    chat_history=None
):
    """
    Creates the prompt that will be
    sent to Gemini.
    """

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for chunk in relevant_chunks:

        context_parts.append(
            f"Source: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Content:\n{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # Build conversation history
    # --------------------------------------------------------

    history_text = ""

    if chat_history:

        for turn in chat_history[-6:]:

            history_text += (
                f"{turn['role'].capitalize()}: "
                f"{turn['content']}\n"
            )


    # --------------------------------------------------------
    # Final prompt
    # --------------------------------------------------------

    prompt = f"""
You are CampusGPT, an AI learning assistant
for students.

IMPORTANT RULES:

1. Answer the question ONLY using the provided
   PDF context.

2. Do NOT use outside knowledge.

3. If the answer cannot be found in the
   provided context, reply exactly:

"I couldn't find that information in the uploaded PDF(s)."

4. When answering, mention the source file
   and page number.

5. Keep the answer clear and educational.

6. Previous conversation may be used only
   when it is relevant to the current question.

--------------------------------------------------
PREVIOUS CONVERSATION
--------------------------------------------------

{history_text}

--------------------------------------------------
PDF CONTEXT
--------------------------------------------------

{context}

--------------------------------------------------
QUESTION
--------------------------------------------------

{question}

--------------------------------------------------
ANSWER
--------------------------------------------------
"""

    return prompt


# ============================================================
# 12. GENERATE ANSWER USING GEMINI
# ============================================================

def generate_answer(prompt):
    response = client.models.generate_content(
        model=GEN_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=512
        )
    )

    if not response.text:
        return "No answer was generated."

    return response.text.strip()

# ============================================================
# 13. CHAT HISTORY
# ============================================================

chat_history = []


# ============================================================
# 14. COMPLETE RAG PIPELINE
# ============================================================

def answer_question(
    question,
    index,
    chunks,
    chat_history,
    verbose=False
):
    """
    Complete RAG pipeline:

    Question
       ↓
    Local embedding
       ↓
    FAISS search
       ↓
    Relevant chunks
       ↓
    Prompt
       ↓
    Gemini
       ↓
    Answer
    """

    # --------------------------------------------------------
    # Step 1: Retrieve relevant chunks
    # --------------------------------------------------------

    relevant_chunks = retrieve_relevant_chunks(
        question,
        index,
        chunks,
        top_k=4
    )


    # --------------------------------------------------------
    # Step 2: Build prompt
    # --------------------------------------------------------

    prompt = build_prompt(
        question,
        relevant_chunks,
        chat_history
    )


    # --------------------------------------------------------
    # Step 3: Generate answer
    # --------------------------------------------------------

    answer = generate_answer(
        prompt
    )


    # --------------------------------------------------------
    # Step 4: Save conversation
    # --------------------------------------------------------

    chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # --------------------------------------------------------
    # Step 5: Display results
    # --------------------------------------------------------

    if verbose:

        print("\n" + "=" * 60)

        print(
            f"QUESTION:\n{question}"
        )

        print("\n" + "-" * 60)

        print(
            f"ANSWER:\n{answer}"
        )

        print("\n" + "-" * 60)

        print("SOURCES USED:")

        for chunk in relevant_chunks:

            print(
                f"- {chunk['source']} "
                f"(Page {chunk['page']}) "
                f"Score: {chunk['score']:.4f}"
            )

        print("=" * 60)


    return answer, relevant_chunks


# ============================================================
# CAMPUSGPT WEB APPLICATION
# ============================================================

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# GLOBAL RAG VARIABLES
# ============================================================

chunks = []
vector_index = None
chat_history = []


# ============================================================
# INITIALIZE RAG WITH DEFAULT PDF
# ============================================================

def initialize_rag(pdf_path):

    global chunks
    global vector_index

    print("\n" + "=" * 60)
    print("Initializing CampusGPT...")
    print("=" * 60)

    # Load PDF
    print("\nLoading PDF...")

    pages_data = load_multiple_pdfs(
        [pdf_path]
    )

    # Create chunks
    print("\nCreating chunks...")

    chunks = chunk_text(
        pages_data,
        chunk_size=150,
        overlap=30
    )

    # Create FAISS index
    print("\nCreating FAISS index...")

    vector_index = build_vector_index(
        chunks
    )

    print("\nCampusGPT RAG initialized successfully!")
    print("=" * 60)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ASK QUESTION
# ============================================================

@app.route("/ask", methods=["POST"])
def ask_question_web():

    global vector_index
    global chunks
    global chat_history

    question = request.form.get(
        "question",
        ""
    ).strip()

    # Check empty question
    if not question:

        return render_template(
            "index.html",
            message="Please enter a question."
        )

    # Check PDF/index
    if vector_index is None or not chunks:

        return render_template(
            "index.html",
            question=question,
            answer="Please upload a PDF first."
        )

    try:

        # Use your existing RAG pipeline
        answer, retrieved_chunks = answer_question(
            question,
            vector_index,
            chunks,
            chat_history,
            verbose=False
        )

        return render_template(
            "index.html",
            question=question,
            answer=answer
        )

    except Exception as e:

        return render_template(
            "index.html",
            question=question,
            answer=f"Error: {str(e)}"
        )


# ============================================================
# PDF UPLOAD
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_pdf():

    global vector_index
    global chunks
    global chat_history

    file = request.files.get("pdf")

    # No file selected
    if file is None or file.filename == "":

        return render_template(
            "index.html",
            message="Please select a PDF file."
        )

    # Only PDF files
    if not file.filename.lower().endswith(".pdf"):

        return render_template(
            "index.html",
            message="Only PDF files are allowed."
        )

    try:

        # Secure filename
        filename = secure_filename(
            file.filename
        )

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        # Save uploaded PDF
        file.save(filepath)

        print(
            f"\nUploaded PDF: {filename}"
        )

        # Load PDF
        pages_data = load_pdf(
            filepath
        )

        # Create chunks
        chunks = chunk_text(
            pages_data,
            chunk_size=150,
            overlap=30
        )

        # Create new FAISS index
        vector_index = build_vector_index(
            chunks
        )

        # Clear old conversation
        chat_history.clear()

        print(
            f"\nSuccessfully processed: {filename}"
        )

        return render_template(
            "index.html",
            message=f"Successfully uploaded: {filename}"
        )

    except Exception as e:

        return render_template(
            "index.html",
            message=f"Error processing PDF: {str(e)}"
        )


# ============================================================
# START WEB SERVER
# ============================================================

if __name__ == "__main__":

    # Load the default sample PDF
    initialize_rag(
        "sample.pdf"
    )

    print("\n" + "=" * 60)
    print("CampusGPT Web App is running!")
    print("Open: http://127.0.0.1:8000")
    print("=" * 60)

    app.run(
        host="127.0.0.1",
        port=8000,
        debug=True
    )