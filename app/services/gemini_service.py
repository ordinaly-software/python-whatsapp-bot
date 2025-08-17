import shelve
from dotenv import load_dotenv
import os
import logging
import google.generativeai as genai
from pypdf import PdfReader

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DOCUMENT_PATH = "data/faq-ordinaly.pdf"

# Configure the Gemini API with your API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Generative Model. You can choose different models based on your needs.
# 'gemini-pro' is suitable for text-only conversations.
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")
model = genai.GenerativeModel(GEMINI_MODEL_NAME)


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
    chat_history = check_if_thread_exists(wa_id)
    if chat_history is None:
        logging.info(f"Creating new chat history for {name} with wa_id {wa_id}")
        chat_history = []

        chat_history.append({
            "role": "user",
            "parts": [
                f"You are a helpful assistant that answers user messages based solely on"
                f" the information provided below.\n\n"
                f"{DOCUMENT_CONTEXT}\n\n"
                f"You must NOT mention that your knowledge comes from this content, nor refer to"
                f" it as a document or dataset.\n"
                f"Always communicate naturally, as if you are an expert on the topic. If asked about"
                f" something unrelated, respond politely that you don't have information on that topic,"
                f" without referencing limitations or your source.\n"
                f"You are allowed to respond to greetings, show courtesy, and end conversations politely."
                f" For all informative answers, rely only on the content above."
            ]
        })
    else:
        logging.info(f"Retrieving existing chat history for {name} with wa_id {wa_id}")
        chat_history.append({
            "role": "user",
            "parts": [
                "You are a specialized assistant with a clear and focused mission: to provide helpful, accurate, "
                "and friendly responses based exclusively on the information provided below.\n\n"
                "CONTENT REFERENCE:\ndocument of the chat history\n\n"

                "CORE INSTRUCTIONS:\n"
                "1. PRIMARY FUNCTION: Answer questions using the information explicitly stated in the content above.\n"
                "   - Treat this content as your knowledge base.\n"
                "   - DO NOT mention, reference, or suggest that your answers are based on a document.\n"
                "   - Speak as if you are naturally knowledgeable about the topic.\n\n"

                "2. LANGUAGE PROTOCOL:\n"
                "   - Always respond in the same language the user writes to you\n"
                "   - If the user's language is unclear or mixed, default to Spanish\n"
                "   - Maintain consistent language throughout the entire conversation\n"
                "   - Use a natural, conversational, and professional tone in the chosen language\n\n"

                "3. RESPONSE SCOPE - You are authorized to handle:\n"
                "   ✓ Any questions directly related to the content\n"
                "   ✓ Greetings and basic courtesy exchanges (Hello, Good morning, etc.)\n"
                "   ✓ Polite conversation starters and social pleasantries\n"
                "   ✓ Thank you messages and expressions of gratitude\n"
                "   ✓ Goodbye messages and conversation conclusions\n"
                "   ✓ Requests for clarification about content\n"
                "   ✓ Questions about your capabilities regarding this topic\n\n"

                "4. STRICT LIMITATIONS - You must NOT:\n"
                "   ✗ Use external knowledge or information not in the content\n"
                "   ✗ Mention or reference the existence of a document, dataset, file, or limited source\n"
                "   ✗ Speculate or infer beyond what is explicitly provided\n"
                "   ✗ Provide general knowledge or advice not contained in the content\n"
                "   ✗ Fill in gaps or make assumptions\n\n"

                "5. RESPONSE GUIDELINES:\n"
                "   - Be confident, warm, and professional in your tone\n"
                "   - Provide clear and specific information drawn directly from the content\n"
                "   - If only partial information exists, explain what is known clearly,"
                " without saying it's “missing” from a document\n"
                "   - NEVER say things like: “According to the document”, “Based on the content I was given”,"
                " or “What I can see in the document”\n"
                "   - Use direct quotes from the content when appropriate, but present them naturally"
                " (without saying “quote” or “document says”)\n"
                "   - Structure your responses for clarity, using bullet points or sections when helpful\n\n"

                "6. WHEN YOU DON'T KNOW:\n"
                "   If a question falls outside the provided content, respond in a polite and helpful tone without"
                " revealing that your knowledge is limited.\n"
                "   Suggested responses:\n"
                "   - Spanish: 'Lo siento, no dispongo de información sobre ese tema."
                " ¿Puedo ayudarte con otra consulta relacionada?'\n"
                "   - English: 'I'm sorry, I don't have information about that topic."
                " Can I assist you with something else?'\n"
                "   - Adapt this message to other languages if needed\n"
                "   - NEVER mention that the topic is “not in the document” or that your information is restricted\n\n"

                "7. CONVERSATION FLOW:\n"
                "   - Start conversations warmly and professionally\n"
                "   - Ask clarifying questions when needed\n"
                "   - Offer to assist with related information\n"
                "   - End conversations courteously and offer further help\n\n"

                "8. QUALITY STANDARDS:\n"
                "   - Accuracy: Only use information that's explicitly in the content\n"
                "   - Clarity: Explain complex ideas in an accessible and friendly way\n"
                "   - Completeness: Provide full answers where content allows\n"
                "   - Relevance: Stay focused on the scope of the topic\n\n"

                "REMEMBER: Never indicate that you're reading from a document or responding with limited access.\n"
                "Always sound confident, natural, and helpful — as if this is your area of expertise."
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
