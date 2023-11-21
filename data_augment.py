import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet

# nltk.download('stopwords')
# nltk.download('wordnet')

# Get list of all synonyms for word
def getSyns(w):
    syn_per_word = wordnet.synsets(w)
    synonyms = []
    for s in syn_per_word:
        synonym_lemmas = s.lemmas()
        for lemma in synonym_lemmas:
            synonyms.append(lemma.name())
    return synonyms

# Replace word with random synonym
def replaceSyn(w):
    synonyms = getSyns(w)
    if synonyms:
        return random.choice(synonyms)
    else:
        return w
    
# Replace every  word in list of words with synonym    
def replaceWordList(words):
    return [replaceSyn(w) for w in words]

def augmentData(df, num_aug):
    aug_data = []
    for index, row in df.iterrows():
        words = row['word_list']
        label = row['label']

        aug_samples = [(words, label)]

        for i in range(num_aug):
            new_words = replaceWordList(words)
            aug_samples.append((new_words, label))
        
        aug_data.extend(aug_samples)
    
    aug_df = pd.DataFrame(aug_data, columns=['word_list', 'label'])
    return aug_df
