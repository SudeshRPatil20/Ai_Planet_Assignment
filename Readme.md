# 🤖 Math Routing Agent: Human-in-the-Loop Educational AI

An open-source, agentic RAG-based assistant that replicates a math professor — solving questions step-by-step using LLMs, vector search, and web search. Feedback from users helps improve the agent over time.

---

## 🎯 Features

- 🔍 Knowledge Base + Web Search Routing + Model Context Protocol
- 🧠 LLM Reasoning
- 🛡️ Input & Output Guardrails
- 🔁 Human-in-the-Loop Feedback (DSPy)
- 🧪 [Bonus] JEE Bench Evaluation

---

## 📹 Demo Video

🎥 [Watch on YouTube](https://www.youtube.com/watch?v=kjMpTZPMxpk)

> Covers:
> - Full architecture explanation
> - Step-by-step demo (KB + web fallback)
> - Guardrails and feedback loop in action

---


# 🤖 Math Routing Agent: Human-in-the-Loop Educational AI

<p align="center">
  <img src="images/Screenshot 2025-08-10 220717.png" width="30%">
  <img src="images/Screenshot 2025-08-10 220537.png" width="30%">
</p>
<p align="center">
  <img src="images/Screenshot 2025-08-10 220555.png" width="30%">
  <img src="images/Screenshot 2025-07-31 161837.png" width="30%">
</p>
<p align="center">
  <img src="Screenshot_2025-08-10_220717.png" width="25%">
</p>

An open-source, agentic RAG-based assistant that replicates a math professor — solving questions step-by-step using LLMs, vector search, and web search. Feedback from users helps improve the agent over time.

---

## 🚀 Key Technologies

| Category               | Technologies Used |
|------------------------|-------------------|
| **Backend** 🖥️        | FastAPI, Python |
| **Frontend** 🎨       | React (TypeScript) |
| **AI & Agents** 🤖    | LangChain, DSPy, Agentic RAG |
| **Vector Database** 📚| FAISS |
| **LLMs** 💡           | Google Gemini (via DSPy), Groq (via LangChain ChatGroq) |
| **Search & Knowledge** 🌐 | Web Search with MCP |
| **Guardrails** 🛡️    | Input/Output Validation |
| **Human-in-the-Loop** 👥 | Feedback-based Fine-tuning (DSPy BootstrapFewShot) |

---

✨ This project combines **real-time problem-solving**, **contextual reasoning**, and **continuous learning** to create an intelligent math tutor that improves with every interaction.



## 🧠 High-Level Design (HLD)

graph TD
    User --> UI[React Frontend]
    UI --> API[FastAPI Backend]
    API --> Guardrails[Input/Output Guardrails]
    Guardrails --> Router[Math Routing Agent]
    Router --> KB[ChromaDB Vector DB]
    Router --> Web[Web Search (DuckDuckGo + MCP)]
    Router --> LLM[LLM (Gemini / HuggingFace)]
    Router --> Feedback[Feedback Engine (DSPy)]
    Feedback --> DB[In-Memory Feedback Store]
⚙️ Low-Level Design (LLD)
mermaid
Copy
Edit
sequenceDiagram
    participant User
    participant Frontend
    participant FastAPI
    participant Guardrails
    participant Router
    participant ChromaDB
    participant WebSearch
    participant LLM
    participant Feedback

    User->>Frontend: Enter math question
    Frontend->>FastAPI: POST /query
    FastAPI->>Guardrails: Validate Input
    Guardrails-->>FastAPI: Pass/Reject
    FastAPI->>Router: Route Query
    Router->>ChromaDB: Search Vector DB
    alt If Match Found
        Router-->>FastAPI: Return KB Result
    else Not Found
        Router->>WebSearch: Search Query
        WebSearch->>Router: Extract Snippets
        Router->>LLM: Generate Answer
    end
    Router->>Guardrails: Validate Output
    Guardrails->>Frontend: Show Answer
    User->>Frontend: Submit Feedback
    Frontend->>FastAPI: POST /feedback
    FastAPI->>Feedback: Learn patterns
🧪 Example Math Questions
🔹 Knowledge Base
What is the derivative of x^2 + 3x + 2?

Solve the quadratic equation x^2 - 5x + 6 = 0

Find the area of a circle with radius 5

🔸 Web Search + MCP
What is the Fourier transform of sin(x)?

Explain eigenvectors and eigenvalues

What is L'Hospital's rule?

🧰 Tech Stack
Layer	Tools
Frontend	React + Tailwind CSS + react-markdown
Backend	FastAPI
Vector DB	ChromaDB + Sentence Transformers
LLMs	Google Gemini 1.5 / HuggingFace DialoGPT
Web Search	DuckDuckGo + BeautifulSoup + MCP
Feedback	DSPy + In-memory pattern store
Diagram Tools	Mermaid (Markdown)

🔌 API Endpoints
POST /query
Submit a math question:

json
Copy
Edit
{
  "question": "solve x^2 + 2x - 8 = 0",
  "user_id": "student123"
}
POST /feedback
Provide feedback on an answer:

json
Copy
Edit
{
  "query_id": "abc123",
  "feedback_type": "positive",
  "rating": 5
}
GET /health
Returns system and model status.

GET /stats
Returns:

Vector count in KB

Feedback stats

🏁 Setup Instructions
Backend Setup
bash
Copy
Edit
pip install -r requirements.txt
uvicorn math_agent:app --reload
Frontend Setup
bash
Copy
Edit
cd frontend
npm install
npm run dev
💡 Human-in-the-Loop Feedback
Collected from /feedback

Learns common question types via positive/negative patterns

Uses DSPy (optional) to fine-tune step-by-step answer generation

🧪 [Bonus] JEE Bench Evaluation
Benchmark your agent’s accuracy and reasoning on the JEE Advanced Math dataset.

Coming soon:

Evaluation script

Auto-grading engine

Result comparison with baseline models

📁 Directory Structure
css
Copy
Edit
.
├── app2.py
├── pd/
│   └── pdf_reader.py
├── sample_data/
│   └── math_qa_dataset_final.pdf
├── frontend/
│   └── [React UI: KaTeX + Markdown]
├── requirements.txt
└── README.md
📄 License
MIT License — Educational use only.

🙌 Acknowledgments
DeepLearning.ai (LangGraph, DSPy, MCP)

OpenAI / Google / HuggingFace

ChromaDB open source

DuckDuckGo Search API

🧑‍💻 Author
Built by Sudesh
📬 sudeshrpatil20121@gmail.com
🌐 GitHub | LinkedIn | Portfolio
