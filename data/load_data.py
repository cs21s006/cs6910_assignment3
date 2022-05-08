import pandas as pd

# Function used to load data from a given path
def load_data(path):
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = ['Devanagari', 'Romanized', 'Attestations']
    df = df.dropna()
    input_texts = df['Romanized'].tolist()
    target_texts = df['Devanagari'].apply(lambda x: 'S' + x + 'E').tolist()
    return input_texts, target_texts