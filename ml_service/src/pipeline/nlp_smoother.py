# nlp_smoother.py - Gemini: fingerspelled ISL words -> English sentence
from src.config import GOOGLE_API_KEY, GEMINI_MODEL

_client = None


def _get_client():
    global _client
    if _client is None and GOOGLE_API_KEY:
        from google import genai
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


def words_to_sentence(words, model_name=GEMINI_MODEL):
    """Turn fingerspelled words (e.g. ['MY', 'NAME', 'RAM']) into one natural
    English sentence. Falls back to joining the words when Gemini is
    unavailable. Blocking - call it off the event loop."""
    if not words:
        return ''

    fallback = ' '.join(words).capitalize()
    client = _get_client()
    if client is None:
        return fallback

    prompt = (
        'A signer fingerspelled these words letter by letter in Indian Sign Language. '
        'Letter recognition is imperfect, so a word may have a swapped, missing, or repeated letter.\n'
        f'Words: {", ".join(words)}\n'
        'Write the single natural English sentence the signer most likely meant, fixing obvious '
        'misspellings. Return only the sentence, without quotes or commentary.'
    )
    try:
        response = client.models.generate_content(model=model_name, contents=prompt)
        sentence = (response.text or '').strip().strip('"\'')
        return sentence or fallback
    except Exception as error:
        print(f'Gemini API error: {error}')
        return fallback
