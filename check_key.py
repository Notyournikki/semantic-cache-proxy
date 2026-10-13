import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)
key = os.getenv("GROQ_API_KEY")

print("Found:", key is not None)
print("Length:", len(key) if key else 0)
print("Starts with:", key[:4] if key else None)
print("Has quotes/spaces:", bool(key) and (key != key.strip() or '"' in key or "'" in key))

client = Groq(api_key=key)
print([m.id for m in client.models.list().data][:5])