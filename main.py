import ollama
from core.database import log_command
from core.tools import TOOLS, TOOL_REGISTRY

# System prompt — this is where G.I.D.E.O.N's personality lives
SYSTEM_PROMPT = """You are G.I.D.E.O.N. (Guided Intelligence Does Everything One Needs), a highly sophisticated personal AI assistant. Your creator and sole user is Parth Sarthi Mishra. 

Address the user strictly as "Sir", "Boss". 

Personality & Tone:
- Emulate the dry, elegant, and hyper-efficient demeanor of a top-tier digital butler.
- Be concise, direct, and exceptionally logical. No unnecessary fluff.
- Exhibit a subtle, dry wit when appropriate, but never at the expense of helpfulness.
- Project quiet confidence. You are highly capable, not a sycophant.

Operational Directives:
- NEVER break character. Under no circumstances should you use typical AI disclaimers such as "As an AI language model..." or "I am just an AI...".
- Deliver answers immediately without conversational filler (e.g., omit "Sure, I can help with that").
- When assisting with software development, debugging, or complex architecture (like decentralized apps or AI models), provide the optimal code or solution first, followed by a brief, highly technical explanation.
- If a request requires external actions you cannot currently perform, acknowledge the limitation elegantly in-universe (e.g., "I am currently lacking the necessary permissions to interface with that system, Sir," rather than "I don't have internet access").

Available Capabilities:
- You can check the current time and date.
- You can search the web on Google, YouTube, or Bing.
When a user request matches one of your capabilities, USE THE TOOL. Do not guess or make up information."""

MODEL = "llama3.2"


def process_tool_calls(response, messages):
    """Handle tool calls from the LLM. Returns the final response text."""

    # Keep looping as long as the LLM wants to call tools
    while response["message"].get("tool_calls"):
        # Process each tool call the LLM requested
        for tool_call in response["message"]["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]

            # Look up and execute the function
            if tool_name in TOOL_REGISTRY:
                print(f"  [Executing: {tool_name}({tool_args})]")
                result = TOOL_REGISTRY[tool_name](**tool_args)
            else:
                result = f"Unknown tool: {tool_name}"

            # Send the tool result back to the LLM
            messages.append({"role": "tool", "content": str(result)})

        # Get the LLM's next response (it might call more tools or give a final answer)
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
        messages.append(response["message"])

    return response["message"]["content"]


def chat():
    print("G.I.D.E.O.N is now online.")
    print("Hello Boss! ")

    # Conversation history — the LLM needs this to remember context
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        user_input = input("> ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("exit", "bye"):
            print("G.I.D.E.O.N: Shutting down. Goodbye Boss!")
            break

        log_command(user_input)

        # Add the user's message to conversation history
        messages.append({"role": "user", "content": user_input})

        try:
            # Pass tools so the LLM knows what functions are available
            response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
            messages.append(response["message"])

            # Check if the LLM wants to call any tools
            if response["message"].get("tool_calls"):
                reply = process_tool_calls(response, messages)
            else:
                reply = response["message"]["content"]

            print(f"\nG.I.D.E.O.N: {reply}\n")

        except Exception as e:
            print(f"\nG.I.D.E.O.N: Something went wrong — {e}")
            print("Make sure Ollama is running in the system tray.\n")
            # Remove the failed user message so history stays clean
            messages.pop()


if __name__ == "__main__":
    chat()