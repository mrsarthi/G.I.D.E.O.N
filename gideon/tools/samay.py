"""G.I.D.E.O.N clock tool ("samay": time)."""

from datetime import datetime

from gideon.tools.base import Param, tool


def get_current_time():
    """Returns the current local time in 12-hour format."""
    now = datetime.now()
    return now.strftime("%I:%M:%S %p")


def get_current_date():
    """Returns the current local date in a human-readable format."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y")


def get_current_datetime():
    """Returns both the current date and time in a human-readable format."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M:%S %p")


# Tool registry: maps subcommand names to their handler functions
SAMAY_TOOLS = {
    "time": get_current_time,
    "date": get_current_date,
    "datetime": get_current_datetime,
}


@tool(
    "get_time_date",
    "Report the current local time and/or date.",
    Param("subcommand", "One of 'time', 'date' or 'datetime'.", default="datetime"),
    # The model gets the clock from the live context block; offering this as
    # a tool only invites a wasted round trip. Kept for CLI/programmatic use.
    exposed=False,
)
def execute(subcommand="datetime"):
    """Main entry point called by the tool dispatcher.

    Args:
        subcommand: One of 'time', 'date', or 'datetime'.

    Returns:
        A string with the result to inject back into the model's context.
    """
    handler = SAMAY_TOOLS.get(subcommand, get_current_datetime)
    return handler()
