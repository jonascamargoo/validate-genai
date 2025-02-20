import nltk
from sentence_transformers import SentenceTransformer, util
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure the VADER lexicon is available
nltk.download('vader_lexicon', quiet=True)

# Initialize required models
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
analyzer = SentimentIntensityAnalyzer()

print("verificando semelhança entre as frases...")

# Define responses
generated_response = "Neymar joga no santos"
print(generated_response)
expected_response = "Neymar não joga no santos"
print(expected_response)

def calculate_similarity(response1, response2):
    """Calculates the similarity between two sentences using text embeddings."""
    emb1 = model.encode(response1, convert_to_tensor=True)
    emb2 = model.encode(response2, convert_to_tensor=True)
    return util.pytorch_cos_sim(emb1, emb2).item()

def analyze_sentiment(response):
    """Returns the compound sentiment score of the response."""
    return analyzer.polarity_scores(response)['compound']

# Calculate similarity
similarity = calculate_similarity(generated_response, expected_response)
print(f"🔍 Porcentagem inicial de similaridade: {similarity * 100:.2f}%")

# Sentiment analysis
generated_sentiment = analyze_sentiment(generated_response)
expected_sentiment = analyze_sentiment(expected_response)

# Adjust similarity if sentiments are opposite
if (generated_sentiment > 0 and expected_sentiment < 0) or (generated_sentiment < 0 and expected_sentiment > 0):
    similarity *= 0.5  # Penalizes similarity for opposite sentiments
    print("⚠️ Sentimentos opostos detectados! Ajustando similaridade.")

# Final result
if similarity > 0.7:
    print("✅ Resposta aprovada!")
else:
    print("❌ Resposta fora do esperado!")
