# app.py

import json
from flask import Flask, request, render_template, jsonify
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

app = Flask(__name__)

# --- BACKEND LOGIC (Your core summarizer, slightly renamed) ---

def summarize_text(text, num_sentences=3):
    """Uses LSA to generate an extractive summary."""
    try:
        # Check if the text is empty or too short
        if not text or len(text.split()) < 5:
            return "Text is too short or empty for effective summarization."

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        
        # Ensure num_sentences is not more than the total sentences available
        num_sentences = min(num_sentences, len(list(parser.document.sentences)))
        
        summary = summarizer(parser.document, num_sentences)
        return " ".join(str(sentence) for sentence in summary)
    
    except Exception as e:
        # Catch any errors during summarization (e.g., if punkt data is missing)
        return f"Error during summarization: {str(e)}"

# --- FLASK ROUTES (API Endpoints) ---

@app.route('/')
def index():
    """Renders the main HTML template."""
    return render_template('index.html')

@app.route('/api/summarize', methods=['POST'])
def summarize_api():
    """Handles the POST request from the frontend and returns the summary."""
    
    # 1. Get the JSON data sent from the frontend
    data = request.get_json()
    
    # 2. Extract text and desired sentence count
    article_text = data.get('text', '')
    num_sentences = data.get('sentences', 3)
    
    # 3. Process the text using your backend function
    summary = summarize_text(article_text, num_sentences)
    
    # 4. Return the result as a JSON object
    return jsonify({
        'summary': summary,
        'status': 'success'
    })

if __name__ == '__main__':
    # You can change the port if needed
    print("Starting Flask server...")
    app.run(debug=True)