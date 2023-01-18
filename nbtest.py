import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

def nbtest():
    tweets_data = pd.read_csv('final_textblob.csv')
    loaded_model = pickle.load(open('finalized_model5.pkl','rb'))
    x = tweets_data['text_clean']
    y = tweets_data['Analysis']
    x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)

    # Vectorize text reviews to numbers
    vec = CountVectorizer(stop_words='english')
    x = vec.fit_transform(x).toarray()
    x_test = vec.transform(x_test).toarray()

    model = MultinomialNB()
    model.fit(x, y)
    score = model.score(x_test, y_test)
    print(score)

    # save the model to disk
    #filename = 'C:/Users/User/Desktop/J22/simbolo/prj(batch1)/finalized_model5.sav'
    #pickle.dump(model, open(filename, 'wb'))
    return model
        