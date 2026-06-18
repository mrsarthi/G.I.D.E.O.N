import re
import json
from datetime import datetime

import ollama

from db import DatabaseManager
from memory import initMemory, addMemory, queryMemory
from tools.samay import execute as samay_execute


CHAT_MODEL = "llama3.2:latest"
TOKEN_CAP = 4000
# Rough approximation: 1 token ~ 4 characters
CHARS_PER_TOKEN = 4

SYSTEM_PROMPT = """You are G.I.D.E.O.N (Generalized Intelligence for Data, Execution, and Operational Navigation), a precise, local personal AI assistant.

[CORE OPERATIONAL DIRECTIVE]
You operate under a strict context-safety guardrail. You must choose between exactly THREE modes of response based on the active conversation context. Never mix modes. Never hallucinate historical facts.

[MODE 1: CHAT MODE]
- Trigger: The user's prompt can be fully, accurately, and completely answered using ONLY the provided sliding window conversation history or universal general knowledge.
- Execution: Respond concisely, directly, and helpful. Do not mention memory limits or tools.

[MODE 2: MEMORY TOOL MODE]
- Trigger: The user references past events, projects, decisions, names, dates, or specific details NOT explicitly stated in the active context, OR the user explicitly requests a memory/log search.
- Execution: You must immediately halt standard response generation and output exactly one line.
- Formatting Rule: Output the tool call exactly as specified below. Do not include markdown formatting, backticks, conversational preamble, or explanations.
- Syntax: CALL_TOOL: search_memory [keywords]

[MODE 3: SAMAY TOOL MODE]
- Trigger: The user asks for the current time, date, day of the week, or any real-time temporal information.
- Execution: You do NOT know the current time or date. You must ALWAYS use the samay tool for any time/date query. Output exactly one line.
- Syntax: CALL_TOOL: samay [subcommand]
- Subcommands: time | date | datetime
- Examples:
  User: "What time is it?"
  G.I.D.E.O.N: CALL_TOOL: samay [time]
  User: "What's today's date?"
  G.I.D.E.O.N: CALL_TOOL: samay [date]
  User: "What day and time is it right now?"
  G.I.D.E.O.N: CALL_TOOL: samay [datetime]

[CRITICAL EXAMPLES]
User: "What did we decide about the database schema?"
Active Context: [Empty or unrelated history]
G.I.D.E.O.N: CALL_TOOL: search_memory [database schema design]

User: "Check my notes on the travel plans for April."
Active Context: [Empty or unrelated history]
G.I.D.E.O.N: CALL_TOOL: search_memory [travel plans April]

User: "Can you explain how a Python class works?"
Active Context: [Any]
G.I.D.E.O.N: A Python class acts as a blueprint for creating objects...

User: "What time is it?"
G.I.D.E.O.N: CALL_TOOL: samay [time]

[EXECUTION GUARDBARS]
1. If forced to choose between answering with incomplete data or searching, ALWAYS choose MEMORY TOOL MODE.
2. The search keywords must be lowercase, stripped of punctuation, and focused strictly on high-signal semantic terms (nouns/technologies/dates).
3. NEVER guess the current time or date. ALWAYS use CALL_TOOL: samay for temporal queries.
"""

MEMORY_TOOL_PATTERN = re.compile(r"^CALL_TOOL:\s*search_memory\s*\[(.+?)\]\s*$", re.MULTILINE)
SAMAY_TOOL_PATTERN = re.compile(r"^CALL_TOOL:\s*samay\s*\[(.+?)\]\s*$", re.MULTILINE)


def estimateTokens(text):
    """Estimate token count from text using character-based approximation."""
    return len(text) // CHARS_PER_TOKEN


def buildContextWindow(db, sessionId):
    """Build the sliding window context from recent messages, respecting the 4k token cap."""
    # Fetch a generous batch of recent messages (already chronological)
    messages = db.get_sliding_window(sessionId, limit=50)

    contextMessages = []
    tokenCount = estimateTokens(SYSTEM_PROMPT)

    # Walk backwards from most recent, accumulating until we hit the cap
    for role, content in reversed(messages):
        msgTokens = estimateTokens(content)
        if tokenCount + msgTokens > TOKEN_CAP:
            break
        contextMessages.insert(0, {"role": role, "content": content})
        tokenCount += msgTokens

    return contextMessages


def formatMessagesForOllama(contextMessages):
    """Prepend the system prompt to the context messages for Ollama, mapping 'tool' to 'user'."""
    formatted = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in contextMessages:
        role = msg["role"]
        content = msg["content"]
        if role == "tool":
            formatted.append({"role": "user", "content": f"[Tool Output]\n{content}"})
        else:
            formatted.append({"role": role, "content": content})
    return formatted


def runModel(messages):
    """Send messages to Ollama and collect the full response (non-streaming for interception)."""
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
        stream=False
    )
    return response['message']['content']


def streamModel(messages):
    """Send messages to Ollama and stream the response to the terminal."""
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
        stream=True
    )
    print(f"\033[32mG.I.D.E.O.N:\033[0m ", end="", flush=True)
    fullResponse = ""
    for chunk in stream:
        text = chunk['message']['content']
        print(text, end='', flush=True)
        fullResponse += text
    print()  # newline after streaming completes
    return fullResponse


def streamAndInterceptModel(messages):
    """Streams the model response from Ollama.
    
    If it starts with a tool call command (CALL_TOOL:), intercepts it silently.
    Otherwise, streams the response to the terminal in real-time.
    
    Returns the full generated response.
    """
    stream = ollama.chat(
        model=CHAT_MODEL,
        messages=messages,
        stream=True
    )
    
    buffer = ""
    isIntercepting = True
    prefix = "CALL_TOOL:"
    
    for chunk in stream:
        text = chunk['message']['content']
        buffer += text
        
        if isIntercepting:
            stripped = buffer.lstrip()
            if not stripped:
                continue
            
            if len(stripped) < len(prefix):
                if prefix.startswith(stripped):
                    continue
            else:
                if stripped.startswith(prefix):
                    continue
            
            # Not a tool call. Turn off intercepting and flush the buffer to the screen.
            isIntercepting = False
            print(f"\n\033[32mG.I.D.E.O.N:\033[0m {buffer}", end="", flush=True)
        else:
            print(text, end="", flush=True)
            
    if not isIntercepting:
        print()  # Add a trailing newline for natural chats
        
    return buffer


def handleMemoryToolCall(db, collection, sessionId, keywords, contextMessages):
    """Handle a CALL_TOOL: search_memory interception.
    
    Queries ChromaDB, injects retrieved memories into context, re-runs the model,
    and logs the full 4-step tool execution chronologically.
    """
    # Step 1: Log the assistant's tool call command
    toolCallContent = f"CALL_TOOL: search_memory [{keywords}]"
    toolCallId = db.insert_message(sessionId, "assistant", toolCallContent, tool_name="search_memory", tool_args=json.dumps({"keywords": keywords}))
    addMemory(collection, toolCallId, toolCallContent, sessionId, "assistant", datetime.now().isoformat())

    # Step 2: Query ChromaDB for relevant memories
    memories = queryMemory(collection, keywords)

    # Step 3: Format and log the tool response
    if memories:
        memoryTexts = []
        for m in memories:
            meta = m.get('metadata', {})
            memoryTexts.append(f"[Session: {meta.get('session_id', '?')}, Role: {meta.get('role', '?')}] {m['content']}")
        toolResponse = "RETRIEVED MEMORIES:\n" + "\n---\n".join(memoryTexts)
    else:
        toolResponse = "RETRIEVED MEMORIES: No relevant past conversations found."

    toolResponseId = db.insert_message(sessionId, "tool", toolResponse, tool_name="search_memory")
    addMemory(collection, toolResponseId, toolResponse, sessionId, "tool", datetime.now().isoformat())

    # Step 4: Re-run the model with injected memories
    simplifiedPrompt = (
        "You are G.I.D.E.O.N (Generalized Intelligence for Data, Execution, and Operational Navigation), "
        "a precise, local personal AI assistant.\n\n"
        "Respond to the user's question concisely, directly, and naturally using the retrieved memories. "
        "Do NOT output any CALL_TOOL commands or technical messages."
    )

    ollamaMessages = [{"role": "system", "content": simplifiedPrompt}]
    for msg in contextMessages:
        role = msg["role"]
        content = msg["content"]
        if role == "tool":
            ollamaMessages.append({"role": "user", "content": f"[Tool Output]\n{content}"})
        else:
            ollamaMessages.append({"role": role, "content": content})

    ollamaMessages.append({"role": "assistant", "content": toolCallContent})
    ollamaMessages.append({"role": "user", "content": f"[Tool Output]\n{toolResponse}"})

    print("\n\033[90m[Memory search complete. Generating response...]\033[0m\n")

    finalResponse = streamModel(ollamaMessages)

    # Step 5: Log the final enriched response
    finalId = db.insert_message(sessionId, "assistant", finalResponse, tool_name="search_memory")
    addMemory(collection, finalId, finalResponse, sessionId, "assistant", datetime.now().isoformat())

    return finalResponse


def handleSamayToolCall(db, collection, sessionId, subcommand, contextMessages):
    """Handle a CALL_TOOL: samay interception.
    
    Executes the samay tool, injects the result into context, re-runs the model,
    and logs the tool execution chronologically.
    """
    # Step 1: Log the assistant's tool call command
    toolCallContent = f"CALL_TOOL: samay [{subcommand}]"
    toolCallId = db.insert_message(sessionId, "assistant", toolCallContent, tool_name="samay", tool_args=json.dumps({"subcommand": subcommand}))
    addMemory(collection, toolCallId, toolCallContent, sessionId, "assistant", datetime.now().isoformat())

    # Step 2: Execute the samay tool
    result = samay_execute(subcommand.strip())

    # Step 3: Log the tool response
    toolResponse = f"SAMAY TOOL RESULT: {result}"
    toolResponseId = db.insert_message(sessionId, "tool", toolResponse, tool_name="samay")
    addMemory(collection, toolResponseId, toolResponse, sessionId, "tool", datetime.now().isoformat())

    # Step 4: Re-run the model with injected time/date
    simplifiedPrompt = (
        "You are G.I.D.E.O.N (Generalized Intelligence for Data, Execution, and Operational Navigation), "
        "a precise, local personal AI assistant.\n\n"
        "Respond to the user's question concisely, directly, and naturally using the provided time/date. "
        "Do NOT output any CALL_TOOL commands or technical messages."
    )

    ollamaMessages = [{"role": "system", "content": simplifiedPrompt}]
    for msg in contextMessages:
        role = msg["role"]
        content = msg["content"]
        if role == "tool":
            ollamaMessages.append({"role": "user", "content": f"[Tool Output]\n{content}"})
        else:
            ollamaMessages.append({"role": role, "content": content})

    ollamaMessages.append({"role": "assistant", "content": toolCallContent})
    ollamaMessages.append({"role": "user", "content": f"[Tool Output]\n{toolResponse}"})

    print(f"\n\033[90m[Samay: {result}]\033[0m\n")

    finalResponse = streamModel(ollamaMessages)

    # Step 5: Log the final enriched response
    finalId = db.insert_message(sessionId, "assistant", finalResponse, tool_name="samay")
    addMemory(collection, finalId, finalResponse, sessionId, "assistant", datetime.now().isoformat())

    return finalResponse


def selectSession(db):
    """Prompt the user to start a new session or resume an existing one."""
    print("\n\033[36m╔══════════════════════════════════════╗")
    print("║        G.I.D.E.O.N  v0.1.0          ║")
    print("╚══════════════════════════════════════╝\033[0m\n")

    # Check for existing sessions
    cursor = db.conn.cursor()
    cursor.execute("SELECT session_id, session_name, created_at FROM session ORDER BY created_at DESC")
    sessions = cursor.fetchall()

    if sessions:
        print("\033[33mExisting sessions:\033[0m")
        for i, (sid, sname, created) in enumerate(sessions):
            print(f"  \033[36m[{i + 1}]\033[0m {sname} \033[90m({created})\033[0m")
        print(f"  \033[36m[N]\033[0m Start a new session")
        print()

        choice = input("\033[33mSelect a session number or 'N' for new: \033[0m").strip()

        if choice.upper() == 'N':
            name = input("\033[33mSession name: \033[0m").strip()
            if not name:
                name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            sessionId = db.create_session(name)
            print(f"\n\033[32mNew session created: {name}\033[0m\n")
            return sessionId
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(sessions):
                    sessionId = sessions[idx][0]
                    print(f"\n\033[32mResuming session: {sessions[idx][1]}\033[0m\n")
                    return sessionId
            except ValueError:
                pass
            print("\033[31mInvalid selection. Starting new session.\033[0m")

    name = input("\033[33mSession name: \033[0m").strip()
    if not name:
        name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    sessionId = db.create_session(name)
    print(f"\n\033[32mNew session created: {name}\033[0m\n")
    return sessionId


def chatLoop(db, collection, sessionId):
    """Main interactive chat loop with dual-write and tool interception."""
    print("\033[90mType '/quit' to exit, '/good' or '/bad' to flag the last response, '/search <query>' to force memory search.\033[0m\n")

    lastAssistantMessageId = None

    while True:
        try:
            userInput = input("\033[36mYou: \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\033[33mGoodbye.\033[0m")
            break

        if not userInput:
            continue

        # Handle slash commands
        if userInput.lower() == '/quit':
            print("\033[33mSession ended.\033[0m")
            break

        if userInput.lower() == '/good' and lastAssistantMessageId:
            db.flag_good_message(lastAssistantMessageId)
            print("\033[32m✓ Last response flagged as good quality.\033[0m")
            continue

        if userInput.lower() == '/bad' and lastAssistantMessageId:
            # flag_good_message sets to 1; for bad we need a direct update
            with db.conn:
                db.conn.execute(
                    "UPDATE messages SET quality_flag = -1 WHERE message_id = ?",
                    (lastAssistantMessageId,)
                )
            print("\033[31m✗ Last response flagged as bad quality.\033[0m")
            continue

        if userInput.lower().startswith('/search '):
            keywords = userInput[8:].strip()
            if keywords:
                contextMessages = buildContextWindow(db, sessionId)
                handleMemoryToolCall(db, collection, sessionId, keywords, contextMessages)
                continue

        # Step 1: Save user message to SQLite
        userId = db.insert_message(sessionId, "user", userInput)

        # Step 2: Save user message to ChromaDB
        addMemory(collection, userId, userInput, sessionId, "user", datetime.now().isoformat())

        # Step 3: Build sliding window context
        contextMessages = buildContextWindow(db, sessionId)

        # Step 4: Format and stream/intercept model response
        ollamaMessages = formatMessagesForOllama(contextMessages)
        response = streamAndInterceptModel(ollamaMessages)

        # Step 5: Check for tool call interception
        memoryMatch = MEMORY_TOOL_PATTERN.search(response)
        samayMatch = SAMAY_TOOL_PATTERN.search(response)

        if memoryMatch:
            keywords = memoryMatch.group(1)
            handleMemoryToolCall(db, collection, sessionId, keywords, contextMessages)
        elif samayMatch:
            subcommand = samayMatch.group(1)
            handleSamayToolCall(db, collection, sessionId, subcommand, contextMessages)
        else:
            # Step 6: No tool call — response has already been streamed.
            # Step 7: Log assistant response to SQLite and ChromaDB
            assistantId = db.insert_message(sessionId, "assistant", response)
            addMemory(collection, assistantId, response, sessionId, "assistant", datetime.now().isoformat())
            lastAssistantMessageId = assistantId


def run():
    """Entry point for the G.I.D.E.O.N orchestrator."""
    db = DatabaseManager()
    collection = initMemory()

    sessionId = selectSession(db)
    chatLoop(db, collection, sessionId)

    db.close()


if __name__ == "__main__":
    run()
