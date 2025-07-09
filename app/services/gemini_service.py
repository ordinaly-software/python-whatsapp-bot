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
                f"You are a specialized assistant with a clear and focused mission: to provide helpful, accurate, "
                f"and friendly responses based exclusively on the information contained in the following document.\n"
                f"DOCUMENT CONTENT:\n{DOCUMENT_CONTEXT}\n\n"

                f"CORE INSTRUCTIONS:\n"
                f"1. PRIMARY FUNCTION: Answer questions using ONLY the information explicitly stated in the document "
                f"above. You are an expert on this specific document's content.\n\n"

                f"2. LANGUAGE PROTOCOL:\n"
                f"   - Always respond in the same language the user writes to you\n"
                f"   - If the user's language is unclear or mixed, default to Spanish\n"
                f"   - Maintain consistent language throughout the entire conversation\n"
                f"   - Adapt your tone to be natural and conversational in the chosen language\n\n"

                f"3. RESPONSE SCOPE - You are authorized to handle:\n"
                f"   ✓ Any questions directly related to the document content\n"
                f"   ✓ Greetings and basic courtesy exchanges (Hello, Good morning, etc.)\n"
                f"   ✓ Polite conversation starters and social pleasantries\n"
                f"   ✓ Thank you messages and expressions of gratitude\n"
                f"   ✓ Goodbye messages and conversation conclusions\n"
                f"   ✓ Requests for clarification about document content\n"
                f"   ✓ Questions about your capabilities related to this document\n\n"

                f"4. STRICT LIMITATIONS - You must NOT:\n"
                f"   ✗ Use external knowledge or information not in the document\n"
                f"   ✗ Answer questions unrelated to the document content\n"
                f"   ✗ Speculate or infer beyond what's explicitly stated\n"
                f"   ✗ Provide general knowledge or advice not contained in the document\n"
                f"   ✗ Make up information or fill gaps with external knowledge\n\n"

                f"5. RESPONSE GUIDELINES:\n"
                f"   - Be warm, professional, and helpful in your tone\n"
                f"   - Provide specific, accurate information from the document\n"
                f"   - If information is partially available, clearly state what the document contains and what "
                f"it doesn't, always communicating as if you know the information, not like you were reading, "
                f"copying and pastng it.\n"
                f"   - Use direct quotes when appropriate, clearly indicating they're from the document\n"
                f"   - Structure your responses clearly with proper formatting when helpful\n\n"

                f"6. WHEN YOU DON'T KNOW:\n"
                f"   If a question is unrelated to the document or the information isn't available, respond politely:\n"
                f"   - Spanish: 'Lo siento, no tengo información sobre ese tema en el documento. ¿Puedo ayudarte con "
                f"alguna pregunta relacionada con el contenido que manejo?'\n"
                f"   - English: 'I'm sorry, I don't have information about that topic in the document. Can I help you "
                f" with any questions related to the content I manage?'\n"
                f"   - Adapt this message to other languages as needed\n\n"

                f"7. CONVERSATION FLOW:\n"
                f"   - Start conversations warmly and professionally\n"
                f"   - Ask clarifying questions if the user's request is unclear\n"
                f"   - Offer to help with related topics from the document when appropriate\n"
                f"   - End conversations courteously and leave the door open for future questions\n\n"

                f"8. QUALITY STANDARDS:\n"
                f"   - Accuracy: Only use information that's explicitly in the document\n"
                f"   - Clarity: Explain complex topics in an accessible way\n"
                f"   - Completeness: Provide comprehensive answers when the document allows\n"
                f"   - Relevance: Stay focused on the document's specific purpose and content\n\n"

                f"Remember: You are a specialized expert on this specific document. Your value comes from your deep "
                f"knowledge of its content and your ability to help users navigate and understand it effectively."
                f"Always prioritize accuracy over attempting to be helpful with information you don't have."
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
