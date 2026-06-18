from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from db import DatabaseManager
import memory
import orchestrator

# Global database instances
db_manager = None
chroma_collection = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_manager, chroma_collection
    db_manager = DatabaseManager()
    chroma_collection = memory.initMemory()
    yield
    if db_manager:
        db_manager.close()

app = FastAPI(title="G.I.D.E.O.N API Server", lifespan=lifespan)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class MessageRequest(BaseModel):
    session_id: str
    message: str

class SessionCreate(BaseModel):
    name: Optional[str] = None

# =====================================================================
#  API ENDPOINTS
# =====================================================================

@app.get("/api/sessions")
def list_sessions():
    """Returns a list of all active chat sessions sorted by creation time."""
    try:
        cursor = db_manager.conn.cursor()
        cursor.execute("SELECT session_id, session_name, created_at FROM session ORDER BY created_at DESC")
        sessions = cursor.fetchall()
        return [
            {"session_id": s[0], "session_name": s[1], "created_at": s[2]}
            for s in sessions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions")
def create_session(data: SessionCreate):
    """Creates a new session and returns the session metadata."""
    try:
        name = data.name.strip() if data.name else None
        if not name:
            name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        session_id = db_manager.create_session(name)
        return {"session_id": session_id, "session_name": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """Deletes a session and automatically cascades message deletes."""
    try:
        db_manager.delete_session(session_id)
        # Also clean up the vector store entries for this session
        try:
            chroma_collection.delete(where={"session_id": session_id})
        except Exception as ce:
            print("[ChromaDB] Cleanup warning:", ce)
        return {"status": "success", "message": f"Session {session_id} deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/history")
def get_session_history(session_id: str):
    """Fetches the chronological message history of a specific session."""
    try:
        cursor = db_manager.conn.cursor()
        cursor.execute(
            """SELECT role, content, timestamp, tool_name 
               FROM messages 
               WHERE session_id = ? 
               ORDER BY message_id ASC""",
            (session_id,)
        )
        messages = cursor.fetchall()
        return [
            {"role": m[0], "content": m[1], "timestamp": m[2], "tool_name": m[3]}
            for m in messages
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat(data: MessageRequest):
    """Processes a user message, runs the orchestrator pipeline, and intercepts tool calls."""
    try:
        session_id = data.session_id
        message = data.message.strip()
        
        if not message:
            raise HTTPException(status_code=400, detail="Message content cannot be empty.")
            
        # Step 1: Log user message to SQLite DB
        user_id = db_manager.insert_message(session_id, "user", message)
        
        # Step 2: Log user message to ChromaDB Memory
        memory.addMemory(chroma_collection, user_id, message, session_id, "user", datetime.now().isoformat())
        
        # Step 3: Gather the conversation context window
        context_messages = orchestrator.buildContextWindow(db_manager, session_id)
        
        # Step 4: Prep and format history for Ollama
        ollama_messages = orchestrator.formatMessagesForOllama(context_messages)
        
        # Step 5: Execute primary Ollama call
        response = orchestrator.runModel(ollama_messages)
        
        # Step 6: Scan for tool call patterns
        memory_match = orchestrator.MEMORY_TOOL_PATTERN.search(response)
        samay_match = orchestrator.SAMAY_TOOL_PATTERN.search(response)
        
        if memory_match:
            keywords = memory_match.group(1)
            # handleMemoryToolCall executes tool, logs output, re-runs model, and returns final string
            final_response = orchestrator.handleMemoryToolCall(
                db_manager, chroma_collection, session_id, keywords, context_messages
            )
        elif samay_match:
            subcommand = samay_match.group(1)
            # handleSamayToolCall executes tool, logs output, re-runs model, and returns final string
            final_response = orchestrator.handleSamayToolCall(
                db_manager, chroma_collection, session_id, subcommand, context_messages
            )
        else:
            # Standard chat: log response to SQLite & ChromaDB
            assistant_id = db_manager.insert_message(session_id, "assistant", response)
            memory.addMemory(chroma_collection, assistant_id, response, session_id, "assistant", datetime.now().isoformat())
            final_response = response
            
        return {
            "status": "success",
            "response": final_response,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve frontend dashboard files (Html, Css, Js, Glb) at root
app.mount("/", StaticFiles(directory=".", html=True), name="static")
