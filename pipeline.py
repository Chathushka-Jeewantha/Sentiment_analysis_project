import  numpy  as np
import pandas as pd 
import re
import string
import pickle

from nltk.stem import PorterStemmer
ps = PorterStemmer()

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)



# load model

with open('static/project/model.pkl', 'rb') as f:
    model=pickle.load(f)
# load stopwords

with open('static/project/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

# load tokens

vocab = pd.read_csv('static/project/vocabulary.txt', header=None)
tokens = vocab[0].tolist()

def remove_punctuations(input_text):
    for punctuation in string.punctuation:
        input_text = input_text.replace(punctuation, '')
    return input_text

def preprocessing(input_text):
    data = pd.DataFrame([input_text], columns=['tweet'])
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data["tweet"] = data['tweet'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
    data["tweet"] = data["tweet"].apply(remove_punctuations)
    data["tweet"] = data["tweet"].str.replace("\\d+", "", regex=True)
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data["tweet"]

def vectorizer(ds, vocabulary):
    vectorized_lst = []
    
    for sentence in ds:
        sentence_lst = np.zeros(len(tokens))
        
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_lst[i] = 1
                
        vectorized_lst.append(sentence_lst)
        
    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)
    
    return vectorized_lst_new

def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    if prediction == 1:
        return 'negative'
    else:
        return 'positive'