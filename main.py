import sys
from config.settings import USER_TITLE
from core.agent import GideonAgent
from core.database import log_command

def main():
    agent = GideonAgent()
    print("G.I.D.E.O.N is now online.")
    print(f"Hello {USER_TITLE}!")

    while True:
        try:
            user_input = input("> ").strip()
            if not user_input: continue
            if user_input.lower() in ("exit", "bye"):
                print(f"G.I.D.E.O.N: Shutting down. Goodbye {USER_TITLE}!")
                break

            log_command(user_input)
            
            print("\nG.I.D.E.O.N: ", end="", flush=True)
            for chunk in agent.stream_chat(user_input):
                print(chunk, end="", flush=True)
            print("\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
