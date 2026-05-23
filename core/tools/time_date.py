from datetime import datetime

def get_current_time() -> str:
    return f"The current time is {datetime.now().strftime('%H:%M')}"

def get_current_date() -> str:
    return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}"
