
# load environment variables from .env file
import os
from dotenv import load_dotenv
key = load_dotenv()
print(f"Loaded .env file: {key}")
print(f"DEEPSEEK_API_KEY: {os.environ.get('DEEPSEEK_API_KEY')}")
