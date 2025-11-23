import json
import nltk  # <--- Added import
from flask import Flask, request, render_template, jsonify
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

app = Flask(__name__)

# --- FIX: Download NLTK data on startup ---
# This MUST be here for the code to work on Render
nltk.download('punkt')
nltk.download('punkt_tab') 

# --- BACKEND LOGIC ---

def summarize_text(text, num_sentences=3):
    """Uses LSA to generate an extractive summary."""
    try:
        if not text or len(text.split()) < 5:
            return "Text is too short or empty for effective summarization."

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        
        num_sentences = min(num_sentences, len(list(parser.document.sentences)))
        
        summary = summarizer(parser.document, num_sentences)
        return " ".join(str(sentence) for sentence in summary)
    
    except Exception as e:
        return f"Error during summarization: {str(e)}"

# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/summarize', methods=['POST'])
def summarize_api():
    data = request.get_json()
    article_text = data.get('text', '')
    num_sentences = data.get('sentences', 3)
    
    summary = summarize_text(article_text, num_sentences)
    
    return jsonify({
        'summary': summary,
        'status': 'success'
    })

if __name__ == '__main__':
    app.run(debug=True)
