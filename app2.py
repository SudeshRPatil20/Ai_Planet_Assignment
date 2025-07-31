# Math Routing Agent - Open Source Implementation
# Free alternatives: Google Gemini, Chroma DB, DuckDuckGo Search, Transformers

import os
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
import logging
from datetime import datetime
import re


# Core dependencies
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
import requests
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import duckduckgo_search as ddg
from bs4 import BeautifulSoup
import numpy as np

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  
    CHROMA_PERSIST_DIR = "./chroma_db"
    COLLECTION_NAME = "math_knowledge_base"
    HUGGINGFACE_MODEL = "microsoft/DialoGPT-medium" 

# Initialize Gemini
if Config.GEMINI_API_KEY:
    genai.configure(api_key=Config.GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-8b-latest')
else:
    logger.warning("GEMINI_API_KEY not found. Using Hugging Face model as fallback.")
    gemini_model = None

# Data Models
class QueryType(Enum):
    KNOWLEDGE_BASE = "knowledge_base"
    WEB_SEARCH = "web_search"
    UNKNOWN = "unknown"

class FeedbackType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

@dataclass
class MathQuery:
    question: str
    user_id: str
    timestamp: datetime
    difficulty_level: Optional[str] = None

@dataclass
class AgentResponse:
    answer: str
    source: str
    confidence: float
    steps: List[str]
    query_type: QueryType
    sources_used: List[str]

class FeedbackRequest(BaseModel):
    query_id: str
    feedback_type: FeedbackType
    feedback_text: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)

# Open Source Guardrails Implementation
class MathGuardrails:
    def __init__(self):
        # Using a simple classification model from Hugging Face
        try:
            self.classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                return_all_scores=True
            )
        except Exception as e:
            logger.warning(f"Could not load classifier: {e}")
            self.classifier = None
    
    def validate_input(self, question: str) -> bool:
        """Check if question is math-related using keyword matching and patterns"""
        math_keywords = [
            'solve', 'equation', 'derivative', 'integral', 'calculate', 'find', 
            'algebra', 'geometry', 'calculus', 'trigonometry', 'probability',
            '+', '-', '*', '/', '=', 'x', 'y', 'z', 'function', 'graph',
            'polynomial', 'matrix', 'vector', 'limit', 'series', 'theorem'
        ]
        
        math_patterns = [
            r'\d+[\+\-\*/]\d+',  # Basic arithmetic
            r'[xy]\^?\d*',       # Variables with powers
            r'sin|cos|tan|log|ln|sqrt',  # Math functions
            r'\d*x[\+\-]\d+',    # Linear equations
            r'∫|∑|π|α|β|γ|θ',    # Math symbols
        ]
        
        question_lower = question.lower()
        
        # Check keywords
        keyword_score = sum(1 for keyword in math_keywords if keyword in question_lower)
        
        # Check patterns
        pattern_score = sum(1 for pattern in math_patterns if re.search(pattern, question_lower))
        
        # Simple scoring system
        total_score = keyword_score + pattern_score * 2
        return total_score >= 2
    
    def validate_output(self, solution: str) -> Dict[str, Any]:
        """Basic validation of mathematical solution"""
        concerns = []
        
       
        has_steps = bool(re.search(r'step\s*\d+|^\d+\.', solution.lower(), re.MULTILINE))
        
    
        has_math = bool(re.search(r'[\+\-\*/=]|\d+|[xy]', solution))
        
      
        is_substantial = len(solution.split()) > 10
        
        if not has_steps:
            concerns.append("Solution lacks clear step-by-step format")
        if not has_math:
            concerns.append("Solution lacks mathematical content")
        if not is_substantial:
            concerns.append("Solution is too brief")
        
        quality = "high" if len(concerns) == 0 else "medium" if len(concerns) <= 1 else "low"
        
        return {
            "is_valid": quality in ["high", "medium"],
            "quality": quality,
            "concerns": "; ".join(concerns)
        }

# Chroma DB Knowledge Base (Free Vector Database)
class KnowledgeBase:
    def __init__(self):
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(path=Config.CHROMA_PERSIST_DIR)
        self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.collection_name = Config.COLLECTION_NAME
        self._setup_collection()
    
    def _setup_collection(self):
        """Initialize Chroma collection"""
        try:
            
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
         
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add math problems and solutions to knowledge base"""
        try:
            questions = [doc["question"] for doc in documents]
            embeddings = self.encoder.encode(questions).tolist()
            
            ids = [f"doc_{i}" for i in range(len(documents))]
            metadatas = [
                {
                    "solution": doc["solution"],
                    "steps": json.dumps(doc.get("steps", [])),
                    "topic": doc.get("topic", "general"),
                    "difficulty": doc.get("difficulty", "medium")
                }
                for doc in documents
            ]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=questions,
                metadatas=metadatas
            )
            logger.info(f"Added {len(documents)} documents to knowledge base")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
    
    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search for similar questions in knowledge base"""
        try:
            query_embedding = self.encoder.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=limit
            )
            
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "question": results['documents'][0][i],
                    "solution": results['metadatas'][0][i]['solution'],
                    "steps": json.loads(results['metadatas'][0][i]['steps']),
                    "score": 1 - results['distances'][0][i],  # Convert distance to similarity
                    "topic": results['metadatas'][0][i].get('topic', 'general')
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Knowledge base search error: {e}")
            return []

# Free Web Search using DuckDuckGo
class WebSearchAgent:
    def __init__(self):
        self.ddg = ddg.DDGS()
    
    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Perform web search using DuckDuckGo"""
        try:
            search_query = f"mathematics {query} step by step solution"
            results = list(self.ddg.text(search_query, max_results=max_results))
            
            return [
                {
                    "title": result.get("title", ""),
                    "content": result.get("body", ""),
                    "url": result.get("href", ""),
                    "relevance_score": 0.8  # Default score since DuckDuckGo doesn't provide one
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []

# Free LLM Handler (Gemini + Hugging Face fallback)
class LLMHandler:
    def __init__(self):
        self.gemini_model = gemini_model
        if not self.gemini_model:
            # Fallback to Hugging Face
            try:
                self.hf_pipeline = pipeline(
                    "text-generation", 
                    model="microsoft/DialoGPT-medium",
                    max_length=100
                )
            except Exception as e:
                logger.error(f"Could not load HuggingFace model: {e}")
                self.hf_pipeline = None
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using available LLM"""
        try:
            if self.gemini_model:
                response = self.gemini_model.generate_content(prompt)
                return response.text
            elif self.hf_pipeline:
                response = self.hf_pipeline(prompt, max_new_tokens=100, num_return_sequences=1) #major changes done
                return response[0]['generated_text']
            else:
                return "I apologize, but I cannot generate a response at the moment. Please check your API configuration."
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Error generating response: {str(e)}"

# Simple Feedback Learning (In-Memory)
class FeedbackLearner:
    def __init__(self):
        self.feedback_storage = []
        self.positive_patterns = []
        self.negative_patterns = []
    
    def add_feedback(self, question: str, answer: str, feedback: Dict[str, Any]):
        """Store feedback for learning"""
        feedback_entry = {
            "question": question,
            "answer": answer,
            "feedback": feedback,
            "timestamp": datetime.now()
        }
        self.feedback_storage.append(feedback_entry)
        
        # Simple pattern learning
        if feedback.get("type") == "positive":
            self.positive_patterns.append(question.lower())
        elif feedback.get("type") == "negative":
            self.negative_patterns.append(question.lower())
    
    def get_confidence_adjustment(self, question: str) -> float:
        """Adjust confidence based on learned patterns"""
        question_lower = question.lower()
        positive_match = any(pattern in question_lower for pattern in self.positive_patterns)
        negative_match = any(pattern in question_lower for pattern in self.negative_patterns)
        
        if positive_match and not negative_match:
            return 0.1
        elif negative_match and not positive_match:
            return -0.1
        return 0.0

# Main Agent State
@dataclass
class AgentState:
    question: str
    user_id: str
    kb_results: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    final_answer: str
    query_type: QueryType
    confidence: float
    steps: List[str]

# Math Routing Agent
class MathRoutingAgent:
    def __init__(self):
        self.llm = LLMHandler()
        self.guardrails = MathGuardrails()
        self.kb = KnowledgeBase()
        self.web_search = WebSearchAgent()
        self.feedback_learner = FeedbackLearner()
    
    def _validate_input(self, state: AgentState) -> AgentState:
        """Validate input using guardrails"""
        if not self.guardrails.validate_input(state.question):
            raise HTTPException(
                status_code=400, 
                detail="Question does not appear to be mathematics-related"
            )
        return state
    
    def _search_knowledge_base(self, state: AgentState) -> AgentState:
        """Search knowledge base for similar questions"""
        state.kb_results = self.kb.search(state.question)
        return state
    
    def _make_routing_decision(self, state: AgentState) -> AgentState:
        """Decide whether to use KB results or search web"""
        if state.kb_results and state.kb_results[0]["score"] > 0.7:
            state.query_type = QueryType.KNOWLEDGE_BASE
            state.confidence = state.kb_results[0]["score"]
        else:
            state.query_type = QueryType.WEB_SEARCH
            state.confidence = 0.5
        
        # Apply feedback learning adjustment
        confidence_adj = self.feedback_learner.get_confidence_adjustment(state.question)
        state.confidence = max(0.0, min(1.0, state.confidence + confidence_adj))
        
        return state
    
    def _web_search(self, state: AgentState) -> AgentState:
        """Perform web search if needed"""
        if state.query_type == QueryType.WEB_SEARCH:
            state.web_results = self.web_search.search(state.question)
        return state
    
    def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate step-by-step solution"""
        if state.query_type == QueryType.KNOWLEDGE_BASE and state.kb_results:
            # Use knowledge base result
            kb_result = state.kb_results[0]
            state.final_answer = kb_result["solution"]
            state.steps = kb_result["steps"]
        else:
            # Generate from web search results or direct reasoning
            context = self._prepare_context(state.web_results) if state.web_results else ""
            
            prompt = f"""
            Solve this mathematics problem step by step:
            
            Question: {state.question}
            
            {f"Additional context: {context}" if context else ""}
            
            Please provide:
            1. A clear understanding of what the problem is asking
            2. Step-by-step solution with explanations
            3. Final answer clearly stated
            
            Format your response with numbered steps.
            """
            
            response = self.llm.generate_response(prompt)
            state.final_answer = response
            state.steps = self._extract_steps(response)
        
        return state
    
    def _prepare_context(self, web_results: List[Dict[str, Any]]) -> str:
        """Prepare context from web search results"""
        if not web_results:
            return ""
        
        context_parts = []
        for result in web_results[:2]:  # Use top 2 results
            content = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
            context_parts.append(f"Source: {result['title']}\n{content}")
        return "\n\n".join(context_parts)
    
    def _extract_steps(self, response: str) -> List[str]:
        """Extract step-by-step solution from response"""
        lines = response.split('\n')
        steps = []
        for line in lines:
            line = line.strip()
            if line and (re.match(r'^\d+\.', line) or line.lower().startswith('step')):
                steps.append(line)
        return steps
    
    def _validate_output(self, state: AgentState) -> AgentState:
        """Validate output using guardrails"""
        validation = self.guardrails.validate_output(state.final_answer)
        if not validation["is_valid"]:
            logger.warning(f"Output validation concerns: {validation['concerns']}")
            # Could implement regeneration logic here
        return state
    
    async def process_query(self, query: MathQuery) -> AgentResponse:
        """Main entry point for processing math queries"""
        try:
            # Initialize state
            state = AgentState(
                question=query.question,
                user_id=query.user_id,
                kb_results=[],
                web_results=[],
                final_answer="",
                query_type=QueryType.UNKNOWN,
                confidence=0.0,
                steps=[]
            )
            
            # Process through pipeline
            state = self._validate_input(state)
            state = self._search_knowledge_base(state)
            state = self._make_routing_decision(state)
            state = self._web_search(state)
            state = self._generate_answer(state)
            state = self._validate_output(state)
            
            return AgentResponse(
                answer=state.final_answer,
                source="knowledge_base" if state.query_type == QueryType.KNOWLEDGE_BASE else "web_search",
                confidence=state.confidence,
                steps=state.steps,
                query_type=state.query_type,
                sources_used=[r.get("url", "") for r in state.web_results]
            )
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def add_feedback(self, query_id: str, feedback: FeedbackRequest):
        """Add human feedback for learning"""
        feedback_data = {
            "type": feedback.feedback_type.value,
            "text": feedback.feedback_text,
            "rating": feedback.rating
        }
        
        
        self.feedback_learner.add_feedback("", "", feedback_data)

# FastAPI Application
app = FastAPI(title="Open Source Math Routing Agent", version="1.0.0")


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)






# Initialize the agent
math_agent = MathRoutingAgent()

# Sample math dataset for knowledge base
# SAMPLE_MATH_DATA = [
#     {
#         "question": "What is the derivative of x^2 + 3x + 2?",
#         "solution": "The derivative is 2x + 3",
#         "steps": [
#             "Step 1: Apply power rule to x^2: d/dx(x^2) = 2x",
#             "Step 2: Apply power rule to 3x: d/dx(3x) = 3", 
#             "Step 3: Derivative of constant 2 is 0",
#             "Step 4: Combine: 2x + 3 + 0 = 2x + 3"
#         ],
#         "topic": "calculus",
#         "difficulty": "easy"
#     },
#     {
#         "question": "Solve the quadratic equation x^2 - 5x + 6 = 0",
#         "solution": "x = 2 or x = 3",
#         "steps": [
#             "Step 1: Factor the quadratic: (x - 2)(x - 3) = 0",
#             "Step 2: Set each factor to zero: x - 2 = 0 or x - 3 = 0",
#             "Step 3: Solve: x = 2 or x = 3"
#         ],
#         "topic": "algebra", 
#         "difficulty": "medium"
#     },
#     {
#         "question": "Find the area of a circle with radius 5",
#         "solution": "The area is 25π or approximately 78.54 square units",
#         "steps": [
#             "Step 1: Use the formula A = πr²",
#             "Step 2: Substitute r = 5: A = π(5)²",
#             "Step 3: Calculate: A = π × 25 = 25π",
#             "Step 4: Approximate: 25π ≈ 78.54 square units"
#         ],
#         "topic": "geometry",
#         "difficulty": "easy"
#     }
# ]

# Initialize knowledge base with sample data
# @app.on_event("startup")
# async def startup_event():
#     math_agent.kb.add_documents(SAMPLE_MATH_DATA)
#     logger.info("Open Source Math Routing Agent initialized successfully")

#changesdone

from pd.pdf_reader import parse_pdf_to_dict  # 👈 Add this at the top
import os

# Initialize knowledge base with PDF data
@app.on_event("startup")
async def startup_event():
    pdf_path = os.path.join("sample_data", "math_qa_dataset_final.pdf")# last changes
    parsed_data = parse_pdf_to_dict(pdf_path)
    math_agent.kb.add_documents(parsed_data)
    logger.info("Knowledge base loaded from PDF successfully")
    
    
    

    
    
    
    
from pydantic import BaseModel


class MathQueryRequest(BaseModel):
    question: str
    user_id: str = "anonymous"
    
    
# API Endpoints
@app.post("/query")
async def process_math_query(
    request: MathQueryRequest
) -> AgentResponse:
    """Process a mathematical query"""
    query = MathQuery(
        question=request.question,
        user_id=request.user_id,
        timestamp=datetime.now()
    )
    return await math_agent.process_query(query)








@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for improving the agent"""
    math_agent.add_feedback(feedback.query_id, feedback)
    return {"message": "Feedback received successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now(),
        "models": {
            "gemini_available": gemini_model is not None,
            "embedding_model": Config.EMBEDDING_MODEL
        }
    }

@app.get("/stats")
async def get_stats():
    """Get agent statistics"""
    return {
        "knowledge_base_count": math_agent.kb.collection.count(),
        "feedback_count": len(math_agent.feedback_learner.feedback_storage),
        "positive_patterns": len(math_agent.feedback_learner.positive_patterns),
        "negative_patterns": len(math_agent.feedback_learner.negative_patterns)
    }

# Installation and setup instructions
"""
Installation Requirements:

1. Install dependencies:
   pip install fastapi uvicorn google-generativeai chromadb sentence-transformers
   pip install transformers torch duckduckgo-search beautifulsoup4 numpy

2. Get free API keys:
   - Google Gemini: https://makersuite.google.com/app/apikey
   - Set environment variable: export GEMINI_API_KEY="your_key_here"

3. Run the application:
   python math_agent.py
   # or
   uvicorn math_agent:app --host 0.0.0.0 --port 8000

4. Test the API:
   curl -X POST "http://localhost:8000/query?question=solve x^2 + 2x - 8 = 0&user_id=test"

Free alternatives used:
- Google Gemini (free tier: 60 requests/minute)
- ChromaDB (open source vector database)
- Sentence Transformers (free embeddings)
- DuckDuckGo Search (free web search)
- Hugging Face Transformers (free models)
"""

# delete


from fastapi import FastAPI
from pydantic import BaseModel



# ✅ Add the health check route here
@app.get("/health")
def health_check():
    return {"status": "ok"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
