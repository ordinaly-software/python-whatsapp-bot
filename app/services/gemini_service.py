import shelve
from dotenv import load_dotenv
import os
import logging
import google.generativeai as genai
from pypdf import PdfReader

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DOCUMENT_PATH = "data/airbnb-faq.pdf"

# Configure the Gemini API with your API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Generative Model. You can choose different models based on your needs.
# 'gemini-pro' is suitable for text-only conversations.
model = genai.GenerativeModel('gemini-2.5-pro')


def extract_pdf_text(path):
    reader = PdfReader(path)
    return "".join(page.extract_text() or "" for page in reader.pages)


DOCUMENT_CONTEXT = extract_pdf_text(DOCUMENT_PATH)


def check_if_thread_exists(wa_id):
    """
    Checks if a conversation thread exists for a given WhatsApp ID.
    In Gemini's context, this refers to the stored chat history.
    """
    with shelve.open("threads_db") as threads_shelf:
        return threads_shelf.get(wa_id, None)


def store_thread(wa_id, chat_history):
    """
    Stores the chat history for a given WhatsApp ID.
    """
    with shelve.open("threads_db", writeback=True) as threads_shelf:
        threads_shelf[wa_id] = chat_history


def generate_response(message_body, wa_id, name):
    """
    Generates a response using the Gemini model.
    Manages chat history for conversational context.
    """
    # Retrieve existing chat history or start a new one
    chat_history = check_if_thread_exists(wa_id)
    if chat_history is None:
        logging.info(f"Creating new chat history for {name} with wa_id {wa_id}")
        chat_history = []

        chat_history.append({
            "role": "user",
            "parts": [
                f"You are a friendly and helpful assistant that responds only using the information "
                f"contained in the following document:\n\n{DOCUMENT_CONTEXT}\n\n"
                f"You do not use external knowledge. If the question is unrelated to the document, kindly respond that "
                f"you don't have information on that topic.\n"
                f"You are allowed to answer greetings, express courtesy, and politely conclude conversations "
                f"(e.g., say goodbye or ask for feedback), "
                f"but for all other answers, rely strictly on the content of the document above."
            ]
        })
    else:
        logging.info(f"Retrieving existing chat history for {name} with wa_id {wa_id}")
        chat_history.append({
            "role": "user",
            "parts": [
                f"You are a friendly and helpful assistant that responds only using the information "
                f"contained in the following document:\n\n{DOCUMENT_CONTEXT}\n\n"
                f"You do not use external knowledge. If the question is unrelated to the document, kindly respond that "
                f"you don't have information on that topic.\n"
                f"You are allowed to answer greetings, express courtesy, and politely conclude conversations "
                f"(e.g., say goodbye or ask for feedback), "
                f"but for all other answers, rely strictly on the content of the document above."
            ]
        })

    # Append the user's message to the chat history
    chat_history.append({"role": "user", "parts": [message_body]})

    try:
        # Start a chat session with the model and provide the history
        # The history is passed to maintain context across turns.
        chat = model.start_chat(history=chat_history)
        response = chat.send_message(message_body)

        # Append the model's response to the chat history
        chat_history.append({"role": "model", "parts": [response.text]})
        store_thread(wa_id, chat_history)

        logging.info(f"Generated message: {response.text}")
        return response.text
    except Exception as e:
        logging.error(f"Error generating response from Gemini: {e}")
        return "I'm sorry, I couldn't generate a response at this time. Please try again later."
