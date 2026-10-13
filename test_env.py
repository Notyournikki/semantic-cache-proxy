from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if api_key:
    print("API key loaded successfully!")
else:
    print("API key was NOT found.")