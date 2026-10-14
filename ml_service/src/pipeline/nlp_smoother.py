# nlp_smoother.py - Gemini API integration for ISL gloss -> English syntax
import os
import google.generativeai as genai
import json
import time

# Configure Gemini API
API_KEY = os.getenv('GOOGLE_API_KEY') or ''
if API_KEY:
    genai.configure(api_key=API_KEY)

DEFAULT_MODEL = 'gemini-1.5-flash'


def smooth_glosses_to_sentence(gloss_sequence, model_name=DEFAULT_MODEL):
    """Convert a sequence of ISL gloss keywords into a natural English sentence.
    
    Args:
        gloss_sequence: List of strings, e.g., ['I', 'STORE', 'GO']
        model_name: Gemini model name to use
    
    Returns:
        String - the grammatically corrected English sentence
    """
    if not gloss_sequence:
        return ''
    
    # Initialize model
    model = genai.GenerativeModel(model_name)
    
    # Format the prompt
    gloss_str = ', '.join(gloss_sequence)
    prompt = f"""You are an expert Indian Sign Language (ISL) translator.
Convert this sequence of ISL gloss keywords into a natural, grammatically correct English sentence.
Return ONLY the final sentence string without quotes, formatting, or extra commentary.

Keywords: {gloss_str}

Final sentence:"""
    
    try:
        response = model.generate_content(prompt)
        sentence = response.text.strip()
        
        # Clean up - remove quotes if present
        if sentence.startswith('"') and sentence.endswith('"'):
            sentence = sentence[1:-1]
        if sentence.startswith("'") and sentence.endswith("'"):
            sentence = sentence[1:-1]
        
        return sentence
    
    except Exception as e:
        print(f'Gemini API error: {e}')
        # Fallback: simple concatenation
        return ' '.join(gloss_sequence)


def batch_translate(gloss_sequences, batch_size=5):
    """Translate multiple gloss sequences in batch.
    
    Args:
        gloss_sequences: List of lists, each containing gloss keywords
        batch_size: Number of sequences to process per batch
    
    Returns:
        List of translated sentences
    """
    results = []
    
    for i in range(0, len(gloss_sequences), batch_size):
        batch = gloss_sequences[i:i+batch_size]
        
        for gloss_seq in batch:
            sentence = smooth_glosses_to_sentence(gloss_seq)
            results.append(sentence)
        
        # Small delay to respect API rate limits
        if i + batch_size < len(gloss_sequences):
            time.sleep(0.5)
    
    return results