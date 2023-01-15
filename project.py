import streamlit as st
from streamlit_lottie import st_lottie
import requests
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
from textblobsa import *
from vadersa import *
from nbtest import *
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
def lottie_codingurl(url):
    animations = requests.get(url)
    if animations.status_code !=200:
        return None
    return animations.json()
lottie_coding = lottie_codingurl('https://assets3.lottiefiles.com/packages/lf20_biekfbpy.json')

@st.cache(persist=True)
def load_data():
    data = pd.read_csv('final_textblob.csv')
    return data
data = load_data()

def load_vader_data():
    data_2 = pd.read_csv('vader.csv')
    return data_2
data_2 = load_vader_data()

    

with st.sidebar:
    st.title('Select Analysis')
    selected = option_menu(menu_title=None,options=['Lexicon','Lexicon+Supervised'],)
	
	
if selected =='Lexicon':
	left_column,right_column = st.columns(2)
	with right_column:
		st_lottie(lottie_coding,height=150,key='coding')
	with left_column:
		st.title('Natural Language Processing with Lexicon')
	contact_options = ['Textblob','VADER']
	selected = st.selectbox('Which sentiment analysis do you want to use?',options=contact_options)
	if selected == 'Textblob':
		text_input = st.text_input('Enter what you want to know the text is Positive, Negative and Neutral')
		clean_text = clean(text_input)
		token_text = token_stop_pos(clean_text)
		lem_text = lemmatize(token_text)
		ploarity_value = getPolarity(lem_text)
		res = analysis(ploarity_value)
		available_for_analysis = st.button('Enter')
		st.write(res)
	else:
		text_input = st.text_input('Enter what you want to know the test is Positive, Negative and Neutral')
		clean_text = clean(text_input)
		token_text = token_stop_pos(clean_text)
		lem_text = lemmatize(token_text)
		ploarity_value = vadersentimentanalysis(lem_text)
		res = vader_analysis(ploarity_value)
		available_for_analysis = st.button('Enter')
		st.write(res)
st.markdown("""---""")
		
        
if selected == 'Lexicon+Supervised':
    left_column,right_column = st.columns(2)
    with right_column:
        st_lottie(lottie_coding,height=150,key='coding')
    with left_column:
        
        st.title('SA with combining Lexicon and Supervised Learning')
    text_input = st.text_input('Enter your text(Textblob Lexicon & MultinomialNB)')
    list = [str(text_input)]
    tweets_data = pd.read_csv('final_textblob.csv')
    x = tweets_data['text_clean']
    y = tweets_data['Analysis']
    x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)

    # Vectorize text reviews to numbers
    vec = CountVectorizer(stop_words='english')
    x = vec.fit_transform(x)
    x_test = vec.transform(x_test)


    model = MultinomialNB()
    model.fit(x, y)
    score = model.score(x_test, y_test)

    # save the model to disk
    filename = 'finalized_model5.sav'
    pickle.dump(model, open(filename, 'wb'))

    # some time later...
    filename = 'finalized_model5.sav' 
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    #loaded_model.score(x_test, y_test)
    score = loaded_model.score(x_test, y_test)
    #vec = CountVectorizer(stop_words='english')
    #result=loaded_model.predict(vec.transform(['str']))
    result=loaded_model.predict(vec.transform(list))
    st.write(result)

    text_input_2 = st.text_input('Enter your text(VADER Lexicon & SVM)')
    tweets_svmdata = pd.read_csv('vader.csv')
    x_train = tweets_svmdata['clean_text']
    y_train = tweets_svmdata['Analysis']
    xs, x_tests, ys, y_tests = train_test_split(x_train,y_train, stratify=y_train, test_size=0.25, random_state=42)
    vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
    train_vectors = vectorizer.fit_transform(xs)
    test_vectors = vectorizer.transform(x_tests)
    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(train_vectors, ys)
    prediction_linear = classifier_linear.predict(vectorizer.transform([text_input_2]))
    
    st.write(prediction_linear)
st.markdown("""---""")


with st.sidebar:
    st.title('Data Set Description')
if not st.sidebar.checkbox("Ignore", True):
    st.title('Data Set Description for US Airline Sentiment Analysis')
    uploaded_file = st.file_uploader('Choose a csv file')
    if uploaded_file is not None:
        df1=pd.read_csv(uploaded_file)
        airline_count = df1.airline.value_counts()
        airline_count = pd.DataFrame({'airlines': airline_count.index, 'AReviews': airline_count.values})
        fig = px.pie(airline_count, values='AReviews', names='airlines')
        st.plotly_chart(fig)
    else:
        st.warning('you need to upload a csv file.')
    st.title('US Airline Sentiment Analysis by Textblob and Vader ')
    select = st.selectbox('Visualization type', ['Bar plot(TextBlob)', 'Pie chart(TextBlob)','Bar plot(VADER)','Pie chart(VADER)'])
    sentiment_count = data['Analysis'].value_counts()
    sentiment_count = pd.DataFrame({'Lexicon_sentiment': sentiment_count.index, 'Reviews': sentiment_count.values})
    sentiment = data_2['Analysis'].value_counts()
    sentiment = pd.DataFrame({'Lexicon_sentiment': sentiment.index,'Reviews': sentiment.values})
    if not st.checkbox("Hide", True):
        st.markdown("## Sentiment Analysis")
        if select == 'Bar plot(TextBlob)':
            fig = px.bar(sentiment_count, x='Reviews',y='Lexicon_sentiment', color='Reviews', orientation="h")
            st.plotly_chart(fig)
        elif select == 'Pie chart(TextBlob)':
            fig = px.pie(sentiment_count,values='Reviews', names='Lexicon_sentiment')
            st.plotly_chart(fig)
        elif select == 'Bar plot(VADER)':
            fig = px.bar(sentiment,x='Reviews',y='Lexicon_sentiment',color='Reviews',orientation='h')
            st.plotly_chart(fig)
        else:
            fig = px.pie(sentiment,values='Reviews',names='Lexicon_sentiment')
            st.plotly_chart(fig)
with st.sidebar:
    st.header('Word Cloud')
    opt = st.sidebar.radio('What sentiment does the word cloud for display?', options=('Positive','Neutral','Negative'))

if not st.sidebar.checkbox("Close", True, key='3'):
    st.subheader('Word Cloud')
    select = st.selectbox('What would you like to see textblob or vader',['Textblob','VADER'])
    if select == 'TextBlob':
        df = data[data['Analysis']==opt]
        words = ' '.join(df['text_clean'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        st.pyplot()
    else:
        df_2 = data_2[data_2['Analysis']==opt]
        words = ' '.join(df_2['clean_text'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
        plt.imshow(wordcloud)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        st.pyplot()