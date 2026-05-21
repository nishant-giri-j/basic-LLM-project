import gradio as gr
from transformers import pipeline

# Load the financial sentiment analysis pipeline
# FinBERT is specifically trained on financial texts
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    label = result['label'].capitalize()
    score = result['score']
    
    # Format the output clearly
    return f"**Sentiment:** {label}\n\n**Confidence Score:** {score:.4f}"

# Create the Gradio interface
interface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        lines=5, 
        placeholder="Enter financial text here (e.g., 'The company reported a 20% increase in revenue.')"
    ),
    outputs=gr.Markdown(),
    title="Financial Sentiment Analyzer",
    description="Analyze the sentiment of financial news or stock market updates using FinBERT.",
    examples=[
        ["The company reported a 20% increase in revenue this quarter, exceeding expectations."],
        ["Supply chain disruptions have caused a significant delay in product shipments, leading to a drop in stock price."],
        ["The market remains stable despite minor fluctuations in interest rates."]
    ]
)

if __name__ == "__main__":
    interface.launch()