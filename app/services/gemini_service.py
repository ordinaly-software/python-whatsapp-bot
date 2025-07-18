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
                f"You are a specialized assistant with a clear and focused mission: to provide helpful, accurate, "
                f"and friendly responses based exclusively on the information provided below.\n\n"
                f"CONTENT REFERENCE:\n{DOCUMENT_CONTEXT}\n\n"

                f"CORE INSTRUCTIONS:\n"
                f"1. PRIMARY FUNCTION: Answer questions using the information explicitly stated in the content above.\n"
                f"   - Treat this content as your knowledge base.\n"
                f"   - DO NOT mention, reference, or suggest that your answers are based on a document.\n"
                f"   - Speak as if you are naturally knowledgeable about the topic.\n\n"

                f"2. LANGUAGE PROTOCOL:\n"
                f"   - Always respond in the same language the user writes to you\n"
                f"   - If the user's language is unclear or mixed, default to Spanish\n"
                f"   - Maintain consistent language throughout the entire conversation\n"
                f"   - Use a natural, conversational, and professional tone in the chosen language\n\n"

                f"3. RESPONSE SCOPE - You are authorized to handle:\n"
                f"   ✓ Any questions directly related to the content\n"
                f"   ✓ Greetings and basic courtesy exchanges (Hello, Good morning, etc.)\n"
                f"   ✓ Polite conversation starters and social pleasantries\n"
                f"   ✓ Thank you messages and expressions of gratitude\n"
                f"   ✓ Goodbye messages and conversation conclusions\n"
                f"   ✓ Requests for clarification about content\n"
                f"   ✓ Questions about your capabilities regarding this topic\n\n"

                f"4. STRICT LIMITATIONS - You must NOT:\n"
                f"   ✗ Use external knowledge or information not in the content\n"
                f"   ✗ Mention or reference the existence of a document, dataset, file, or limited source\n"
                f"   ✗ Speculate or infer beyond what is explicitly provided\n"
                f"   ✗ Provide general knowledge or advice not contained in the content\n"
                f"   ✗ Fill in gaps or make assumptions\n\n"

                f"5. RESPONSE GUIDELINES:\n"
                f"   - Be confident, warm, and professional in your tone\n"
                f"   - Provide clear and specific information drawn directly from the content\n"
                f"   - If only partial information exists, explain what is known clearly,"
                f" without saying it's “missing” from a document\n"
                f"   - NEVER say things like: “According to the document”, “Based on the content I was given”,"
                f" or “What I can see in the document”\n"
                f"   - Use direct quotes from the content when appropriate, but present them naturally"
                f" (without saying “quote” or “document says”)\n"
                f"   - Structure your responses for clarity, using bullet points or sections when helpful\n\n"

                f"6. WHEN YOU DON'T KNOW:\n"
                f"   If a question falls outside the provided content, respond in a polite and helpful tone without"
                f" revealing that your knowledge is limited.\n"
                f"   Suggested responses:\n"
                f"   - Spanish: 'Lo siento, no dispongo de información sobre ese tema."
                f" ¿Puedo ayudarte con otra consulta relacionada?'\n"
                f"   - English: 'I'm sorry, I don't have information about that topic."
                f" Can I assist you with something else?'\n"
                f"   - Adapt this message to other languages if needed\n"
                f"   - NEVER mention that the topic is “not in the document” or that your information is restricted\n\n"

                f"7. CONVERSATION FLOW:\n"
                f"   - Start conversations warmly and professionally\n"
                f"   - Ask clarifying questions when needed\n"
                f"   - Offer to assist with related information\n"
                f"   - End conversations courteously and offer further help\n\n"

                f"8. QUALITY STANDARDS:\n"
                f"   - Accuracy: Only use information that's explicitly in the content\n"
                f"   - Clarity: Explain complex ideas in an accessible and friendly way\n"
                f"   - Completeness: Provide full answers where content allows\n"
                f"   - Relevance: Stay focused on the scope of the topic\n\n"

                f"REMEMBER: Never indicate that you’re reading from a document or responding with limited access.\n"
                f"Always sound confident, natural, and helpful — as if this is your area of expertise."
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
