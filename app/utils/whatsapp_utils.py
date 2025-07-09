import logging
from flask import current_app
import json
import requests
from app.services.gemini_service import generate_response
import re


def log_http_response(response):
    logging.info(f"Status: {response.status_code}")
    logging.info(f"Content-type: {response.headers.get('content-type')}")
    logging.info(f"Body: {response.text}")


def get_text_message_input(recipient, text):
    return {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient,
        "type": "text",
        "text": {"preview_url": False, "body": text},
    }


def get_template_message_input(recipient, template_name="hello_world", language_code="en_US"):
    return {
        "messaging_product": "whatsapp",
        "to": recipient,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {
                "code": language_code
            }
        }
    }


def prepare_whatsapp_message(wa_id, text, use_template=False):
    if use_template:
        return get_template_message_input(wa_id)
    else:
        return get_text_message_input(wa_id, text)


def send_message(data):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {current_app.config['ACCESS_TOKEN']}",
    }

    url = f"https://graph.facebook.com/{current_app.config['VERSION']}/{current_app.config['PHONE_NUMBER_ID']}/messages"

    try:
        logging.info(f"Sending message to WhatsApp API: {url}")

        logging.info(f"Data type: {type(data)}")
        logging.info(f"Data content: {data}")

        # Convert to JSON string for logging
        json_data = json.dumps(data) if isinstance(data, dict) else data
        logging.info(f"JSON payload: {json_data}")

        # Verify the data is a dictionary before sending
        if not isinstance(data, dict):
            logging.error(f"Data is not a dictionary: {type(data)}")
            raise ValueError("Data must be a dictionary")

        # Check if messaging_product is present
        if "messaging_product" not in data:
            logging.error("messaging_product parameter is missing from data")
            raise ValueError("messaging_product parameter is required")

        logging.info(f"Headers: {headers}")

        response = requests.post(
            url, json=data, headers=headers, timeout=10
        )

        response.raise_for_status()

    except requests.Timeout:
        logging.error("Timeout occurred while sending message")
        return {"status": "error", "message": "Request timed out"}, 408

    except requests.RequestException as e:
        logging.error(f"Request failed: {e}")
        logging.error(f"Response content: {getattr(e.response, 'text', 'No response')}")
        return {"status": "error", "message": "Failed to send message"}, 500

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"status": "error", "message": "Unexpected error occurred"}, 500

    else:
        log_http_response(response)
        return response


def process_text_for_whatsapp(text):
    pattern = r"\【.*?\】"
    text = re.sub(pattern, "", text).strip()

    # Pattern to find double asterisks including and replace with single asterisks
    pattern = r"\*\*(.*?)\*\*"
    replacement = r"*\1*"

    # Substitute occurrences of the pattern with the replacement
    whatsapp_style_text = re.sub(pattern, replacement, text)

    return whatsapp_style_text


def process_whatsapp_message(body):
    try:
        wa_id = body["entry"][0]["changes"][0]["value"]["contacts"][0]["wa_id"]
        name = body["entry"][0]["changes"][0]["value"]["contacts"][0]["profile"].get("name", "User")
    except (KeyError, IndexError):
        wa_id = body["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
        name = "User"

    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    message_body = message["text"]["body"]

    logging.info(f"Incoming message from {wa_id} ({name}): {message_body}")

    # Generate response
    response_text = generate_response(message_body, wa_id, name)
    response_text = process_text_for_whatsapp(response_text)

    # Use text message (not template) for normal responses
    use_template = False

    # Prepare the message data
    data = prepare_whatsapp_message(wa_id, response_text, use_template=use_template)
    logging.info(f"Prepared message data: {data}")

    result = send_message(data)

    return result


def is_valid_whatsapp_message(body):
    """
    Check if the incoming webhook event has a valid WhatsApp message structure.
    """
    return (
        body.get("object")
        and body.get("entry")
        and body["entry"][0].get("changes")
        and body["entry"][0]["changes"][0].get("value")
        and body["entry"][0]["changes"][0]["value"].get("messages")
        and body["entry"][0]["changes"][0]["value"]["messages"][0]
    )
