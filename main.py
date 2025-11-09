import core.gideon

print("""G.I.D.E.O.N is now online.
Hello Boss!""")

def main():
    while True:
        userInput = input("You: ")
        if userInput.strip().lower() == "exit":
            print("G.I.D.E.O.N: Shutting down. Goodbye Boss!")
            break

        elif userInput.strip().lower() == "what time is it":
            print(core.gideon.samay())

        else:
            print("G.I.D.E.O.N: I'm sorry, I didn't understand that command.")


if __name__ == "__main__":
    main()
        