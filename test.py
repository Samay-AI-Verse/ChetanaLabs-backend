import requests
import json

# Fill in your details here directly in the code
API_KEY = "baaf291a-563b-43b3-93ff-530839fd1b1d"  # Replace with your actual Vapi API key
PHONE_NUMBER_ID = "59d31a45-4c10-48c7-8112-02b872259ab4"  # Your outbound phone number ID from Vapi dashboard
TARGET_PHONE_NUMBER = "+919764096358"  # The candidate's phone number in E.164 format (e.g., +11234567890)

# Optional: If you have a pre-created assistant ID, fill it here; otherwise leave as None to use transient assistant
ASSISTANT_ID = "2128baf8-b20b-4eae-a219-c69e81e08241"  # e.g., "assistant-uuid-here" or None

# Transient assistant configuration (used if ASSISTANT_ID is None)
# You can customize these
MODEL = "gpt-4o"  # Or other supported models like gpt-4-turbo
VOICE_PROVIDER = "elevenlabs"  # Common providers: elevenlabs, azure, playht
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # e.g., "EXAVITQu4vr4xnSDxMaL" (Rachel or any voice you like)

# Simple HR system prompt
SYSTEM_PROMPT = """
You are a professional HR assistant from [Your Company Name].
You are calling to follow up on a job application for the position of Software Engineer.
Start by greeting and confirming the person's name.
Ask if this is a good time to talk briefly.
Confirm their interest in the role.
If interested, ask about their availability for a short interview in the coming days.
Be polite, concise, and natural. Speak slowly and clearly.
If they are not interested or busy, thank them and end the call gracefully.
Do not ramble or repeat unnecessarily.
"""

FIRST_MESSAGE = "Hello, this is an HR assistant from [Your Company Name]. Am I speaking with [Candidate Name]?"

def make_outbound_call():
    url = "https://api.vapi.ai/call"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    if ASSISTANT_ID:
        # Use pre-created assistant
        payload = {
            "assistantId": ASSISTANT_ID,
            "phoneNumberId": PHONE_NUMBER_ID,
            "customer": {
                "number": TARGET_PHONE_NUMBER
            }
        }
    else:
        # Use transient assistant with custom config
        payload = {
            "assistant": {
                "firstMessage": FIRST_MESSAGE,
                "model": {
                    "provider": "openai",
                    "model": MODEL
                },
                "voice": {
                    "provider": VOICE_PROVIDER,
                    "voiceId": VOICE_ID
                },
                "systemPrompt": SYSTEM_PROMPT
            },
            "phoneNumberId": PHONE_NUMBER_ID,
            "customer": {
                "number": TARGET_PHONE_NUMBER
            }
        }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 201 or response.status_code == 200:
        print("Call initiated successfully!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    make_outbound_call()