from core.tools.time_date import get_current_time, get_current_date
from core.tools.web import web_search
from core.tools.media import play_music

TOOL_REGISTRY = {
    "get_current_time": get_current_time,
    "get_current_date": get_current_date,
    "web_search": web_search,
    "play_music": play_music,
}

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': 'Get the current time.',
            'parameters': {'type': 'object', 'properties': {}},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_current_date',
            'description': 'Get today\'s date.',
            'parameters': {'type': 'object', 'properties': {}},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'web_search',
            'description': 'Search the web.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'platform': {'type': 'string', 'enum': ['google', 'youtube', 'bing', 'music']},
                },
                'required': ['query', 'platform'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'play_music',
            'description': 'Play music or video on YouTube.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                },
                'required': ['query'],
            },
        },
    },
]
