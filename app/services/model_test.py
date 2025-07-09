from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure the Gemini API with your API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Generative Model. You can choose different models based on your needs.
# 'gemini-pro' is suitable for text-only conversations.
models = genai.list_models()
for m in models:
    print(m.name, m.supported_generation_methods)
