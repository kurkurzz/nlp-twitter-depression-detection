import streamlit as st
import pandas as pd

import twint

def scrape_tweet(username):
	c = twint.Config()

	c.Username =  username
	c.Limit = 100
	c.Stats = True
	c.Pandas = True 
	c.Hide_output = False
	# c.Store_csv = True
	# c.Output = 'tweets.csv'
	twint.run.Search(c)

st.title('Twitter Depression Detection.')
st.caption("")
username = st.text_input('Please insert twitter username in the field below.')

df = pd.DataFrame()
if st.button('Submit', key=None):
	scrape_tweet(username)
	df = twint.storage.panda.Tweets_df
	
	df = df[['tweet', 'hashtags']]
	st.dataframe(df)

