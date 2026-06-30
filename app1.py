import gradio as gr
import pandas as pd
import torch
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import matplotlib.pyplot as plt

# ====================== NLTK SETUP ======================
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    punct_to_remove = string.punctuation.replace("'","").replace('"',"").replace("$","").replace("%","").replace("?","")
    text = re.sub(f"[{punct_to_remove}]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

classifier_model = "Ginidu2003/Distilbert-Base-News-classifier"

# ====================== BEAUTIFUL COLORED BAR CHART ======================
def create_colored_bar_chart(category_counts):
    if category_counts is None or len(category_counts) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig

    categories = category_counts["Category"]
    counts = category_counts["Count"]

    # Nice modern color palette
    colors = ['#3498DB', '#E67E22', '#9B59B6', '#2ECC71', '#E74C3C']

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(categories, counts, color=colors, edgecolor='white', linewidth=0.8)

    # Add value on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.8,
                str(int(height)), ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_title("Category Distribution Across 5 Classes", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig

# ====================== CLASSIFICATION FUNCTION ======================
@torch.no_grad()
def classify_csv(file):
    try:
        df = pd.read_csv(file)
        if 'content' not in df.columns:
            return "Error: CSV must have a column named 'content'", None, None
        
        df['clean_content'] = df['content'].apply(preprocess_text)
        
        classifier = pipeline("text-classification", model=classifier_model, device=-1)
        
        predictions = []
        for text in df['clean_content']:
            if not text.strip():
                predictions.append("Unknown")
            else:
                result = classifier(text)[0]
                predictions.append(result['label'])
        
        df['class'] = predictions
        df = df.drop(columns=['clean_content'], errors='ignore')
        
        output_file = "output.csv"
        df.to_csv(output_file, index=False)
        
        category_counts = df['class'].value_counts().reset_index()
        category_counts.columns = ["Category", "Count"]
        
        fig = create_colored_bar_chart(category_counts)
        
        return f"✅ Success! Classified {len(df)} rows", output_file, fig
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None

# ====================== Q&A FUNCTION ======================
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

def answer_question(news_content, question):
    if not news_content.strip() or not question.strip():
        return "Please enter both news content and a question."
    try:
        inputs = qa_tokenizer(question, news_content, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = qa_model(**inputs)
        
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1
        
        answer = qa_tokenizer.decode(inputs.input_ids[0][start_idx:end_idx],
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)
        
        confidence = torch.max(torch.softmax(outputs.start_logits, dim=1)).item()
        
        return f"**Answer:** :-  {answer.strip()}\n\n**Confidence:** :-  {confidence:.2%}"
    except Exception as e:
        return f"Error: {str(e)}"

# ====================== BEAUTIFUL UI ======================
with gr.Blocks(
    title="News Classifier & Question Answering App",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1250px;
        min-width: 1000px;
        margin: auto;
        background: linear-gradient(135deg, #0f172a 0%, #1e2937 100%);
    }
    h1 {
        text-align: center;
        font-size: 2.9rem;
        background: linear-gradient(90deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    /* Upload Box - Gradient */
    .file-upload {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        border-radius: 16px;
        border: none;
    }
    /* Classify News Button - Gradient */
    button.primary {
        background: linear-gradient(90deg, #6366f1, #a855f7) !important;
        border: none;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 14px 0;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    button.primary:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 25px rgba(139, 92, 246, 0.5);
    }
    /* Status Box - Gradient */
    .status-box {
        background: linear-gradient(135deg, #10b981, #34d399) !important;
        color: white;
        border-radius: 12px;
    }
    /* Download Button - Gradient */
    button.secondary {
        background: linear-gradient(90deg, #ec4899, #f43f5e) !important;
        color: white;
        font-weight: 600;
    }
    /* Tab styling */
    .tab-label {
        font-size: 1.15rem;
        font-weight: 600;
    }
    """
) as demo:

    gr.Markdown("# 📰 News Classifier & Question Answering App..")
    

    with gr.Tabs():
        with gr.Tab("📊 News Classification"):
            gr.Markdown("### Upload CSV and get automatic category prediction")
            
            file_input = gr.File(
                label="📤 Upload your CSV file",
                file_types=[".csv"],
                height=160
            )
            
            classify_btn = gr.Button("🚀 Classify News", variant="primary", size="large")
            
            with gr.Row():
                output_text = gr.Textbox(label="Status", scale=2)
                output_file = gr.File(label="📥 Download output.csv")
            
            bar_chart = gr.Plot(label="📊 Category Distribution Across 5 Classes")

            classify_btn.click(
                fn=classify_csv,
                inputs=file_input,
                outputs=[output_text, output_file, bar_chart]
            )

        with gr.Tab("❓ Question Answering"):
            gr.Markdown("### Ask any question about a news article")
            news_input = gr.Textbox(lines=12, label="📝 Paste News Content", placeholder="Paste the full news article here...")
            question_input = gr.Textbox(label="❓ Your Question", placeholder="e.g. What is the main topic?")
            qa_btn = gr.Button("🔍 Get Answer", variant="primary", size="large")
            qa_output = gr.Textbox(label="💡 Answer", lines=6)

            qa_btn.click(
                fn=answer_question,
                inputs=[news_input, question_input],
                outputs=qa_output
            )

    gr.Markdown("---")
    

demo.launch()