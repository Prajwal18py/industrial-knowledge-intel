import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = None
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_groq_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=api_key)
    return _client
