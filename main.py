import core.actions
import time

print("""G.I.D.E.O.N is now online.
Hello Boss!""")

def queryParse(input):
    command = input.strip().lower()
    words = command.split()
    platform = None
    query = None
    if len(words) > 1 and words[0] == "open":
        platform = words[1]
    for phrase in ["search for", "search", "find"]:
        if phrase in command:
            query = command.split(phrase, 1)[1].strip()
            break
    return platform, query


def commandRouter():
    while True:
        userInput = input("> ").strip().lower()

        if userInput == "exit" or userInput == "bye":
            print("G.I.D.E.O.N: Shutting down. Goodbye Boss!")
            break

        elif userInput == "what time is it":
            print(core.actions.samay())

        elif userInput == "what day is it":
            print(core.actions.din())

        elif userInput.startswith("open"):
            platform, query = queryParse(userInput)
            print(f"G.I.D.E.O.N: Opening {platform} and searching for '{query}'")
            time.sleep(2)
            print(core.actions.browser(query, platform))
            # print(platform, query)

        else:
            print("G.I.D.E.O.N: I'm sorry, I didn't understand that command.")


if __name__ == "__main__":
    commandRouter()


# open youtube and search for nlu tutorials
