# 🎓 CampusGPT — AI Learning Assistant

CampusGPT is an AI-powered learning assistant that allows students to upload PDF documents and ask questions about their content.

The system uses ***Retrieval-Augmented Generation (RAG)*** to retrieve relevant information from uploaded PDF documents and generate answers using Google's Gemini AI model.

---

## 📌 Project Overview

Students often need to search through lengthy study materials, notes, textbooks, and other PDF documents to find specific information.

CampusGPT provides a simple web-based interface where users can:

- Upload PDF documents
- Ask questions about the uploaded documents
- Retrieve relevant information from the documents
- Generate AI-based answers
- View source and page information
- Continue asking questions through the chat interface

The main objective of the project is to provide a fast and user-friendly AI learning assistant based on the user's uploaded study materials.

---

# 🎯 Objectives

The main objectives of CampusGPT are:

1. To provide an interactive AI-based learning assistant.
2. To allow users to upload PDF documents.
3. To extract text from PDF documents automatically.
4. To divide large documents into smaller text chunks.
5. To convert text chunks into numerical embeddings.
6. To store and search embeddings using FAISS.
7. To retrieve the most relevant document content for a user's question.
8. To generate answers using Google's Gemini AI model.
9. To provide answers based only on the uploaded document context.
10. To display source information for better answer verification.

---

# ✨ Features

## 📄 PDF Upload

Users can upload PDF documents directly through the web interface.

The application extracts text from the uploaded document and prepares it for searching.

---

## 🔎 Semantic Search

CampusGPT uses semantic similarity rather than simple keyword matching.

The user's question is converted into an embedding and compared with document embeddings using FAISS.

This helps the system find relevant information even when the wording of the question is different from the wording in the document.

---

## 🤖 AI Answer Generation

After retrieving relevant document sections, CampusGPT sends the retrieved context to the Gemini model.

Gemini generates a natural-language response using the retrieved information.

---

## 📚 Source Citations

The system is designed to provide source information such as:

- PDF filename
- Page number

This allows users to verify the information from the original document.

---

## 💬 Interactive Chat Interface

The application provides a web-based chat interface where users can enter questions and receive AI-generated answers.

---

## 🧠 Retrieval-Augmented Generation

CampusGPT follows a Retrieval-Augmented Generation architecture.

Instead of asking the AI model to answer from general knowledge alone, the system first searches the uploaded document for relevant information.

The retrieved information is then provided to the AI model as context.

---

# 🏗️ System Architecture

The overall system follows this pipeline:

```text
                 ┌───────────────────┐
                 │    User Uploads   │
                 │       PDF         │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │    PDF Text       │
                 │    Extraction     │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   Text Chunking   │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │    Sentence      │
                 │    Transformer   │
                 │    Embeddings    │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   FAISS Vector    │
                 │      Index        │
                 └─────────┬─────────┘
                           │
                           │
             User Question
                    │
                    ▼
           ┌───────────────────┐
           │ Question Embedding│
           └─────────┬─────────┘
                     │
                     ▼
           ┌───────────────────┐
           │ FAISS Similarity   │
           │     Search         │
           └─────────┬─────────┘
                     │
                     ▼
           ┌───────────────────┐
           │ Relevant Document │
           │     Context       │
           └─────────┬─────────┘
                     │
                     ▼
           ┌───────────────────┐
           │   Gemini AI       │
           │ Answer Generation │
           └─────────┬─────────┘
                     │
                     ▼
           ┌───────────────────┐
           │     CampusGPT     │
           │      Answer       │
           └───────────────────┘