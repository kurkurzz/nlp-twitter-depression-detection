import datetime as dt
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import twint

from processing import load_models, predict_pandas, preprocess

def scrape_tweet(username):
	c = twint.Config()

	c.Username =  username
	c.Limit = 500
	c.Stats = True
	c.Pandas = True 
	c.Hide_output = False
	today_date = dt.datetime.now()
	c.Until = today_date.strftime(r'%Y-%m-%d')
	last_3month_date = today_date - dt.timedelta(days=90)
	c.Since = last_3month_date.strftime(r'%Y-%m-%d')
	c.Filter_retweets = True
	# c.Store_csv = True
	# c.Output = 'tweets.csv'
	twint.run.Search(c)

st.title('Twitter Depression Detection.')
st.caption("")
username = st.text_input('Please insert twitter username in the field below.')

df = pd.DataFrame()
if st.button('Submit', key=None):
	try:
		username = username.replace('@', '')
		with st.spinner('Analysing your tweets...'):
			scrape_tweet(username)
			df = twint.storage.panda.Tweets_df
			
			df = df[['tweet', 'hashtags']]
			vectoriser, LRmodel = load_models()
			df['cleaned_tweet'] = df['tweet'].apply(preprocess)
			df['proba'] = df['cleaned_tweet'].apply(lambda x: predict_pandas(vectoriser, LRmodel, x))
			df['label'] = df['proba'].apply(lambda x: 1 if x[1] >= 0.5 else 0)
			total_positive = df['label'].sum()
			overall_sentiment = df['proba'].mean()[0]

			y = np.array([total_positive, len(df) - total_positive])
			labels = ['Positive', 'Negative']
			explode = (0, 0.1)
			fig, ax = plt.subplots()
			ax.pie(y, labels=labels, startangle=90, 
					textprops={'color':"w"}, autopct='%1.0f%%',
					explode=explode, colors=['green', 'red'])
		st.header('Result')
		st.pyplot(fig, transparent=True)
		st.markdown(f'### We analyzed <span style="color:yellow">{len(df)}</span> of your tweets from the past 3 months.', unsafe_allow_html=True)
		st.markdown(f'### The result shows that <span style="color:red">{(len(df) - total_positive)/len(df)*100:.0f}%</span> of the tweets are <span style="color:red">negative</span>.', unsafe_allow_html=True)
		# st.markdown(f'### You overall sentiment is: {overall_sentiment}')
		# st.progre()
		if (len(df) - total_positive)/len(df) >= 0.5:
			st.error(f'You are showing depressive symptom and we recommend you to consult with a therapist for a check up and diagnosis.')
		else:
			st.success(f'Your mental health is probably in a good condition. Keep it up and take care of yourself always.')
	except Exception as e:
		st.exception(e)

st.caption('''Disclaimer: This should not be used to diagnose mental health. 
Mental health is a disease and only could be diagnose by a doctor.''')