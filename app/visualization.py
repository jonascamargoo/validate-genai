import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_distribution(df):
    plt.figure(figsize=(8, 5))
    sentiment_counts = df['sentiment'].map({0: 'Negativo', 1: 'Positivo', 2: 'Neutro'}).value_counts()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Distribuição de Sentimentos')
    plt.ylabel('Contagem')
    plt.xlabel('Sentimento')
    plt.tight_layout()
    plt.show()
