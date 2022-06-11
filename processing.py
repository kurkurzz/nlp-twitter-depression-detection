import re
import pickle

# Natural Language Toolkit (nltk)
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
			':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
			':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
			':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
			'@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
			'<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
			';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

mystopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
				'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
				'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
				'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
				'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
				'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
				'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
				'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
				'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
				's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
				't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
				'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
				'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
				'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
				'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
				"youve", 'your', 'yours', 'yourself', 'yourselves']

stopwordlist = stopwords.words('english') + mystopwordlist

def preprocess(tweet):
	#creating a Lemmatizer
	wordLemma = WordNetLemmatizer() #define the imported library

	# Defining regular expression pattern we can find. in tweets

	urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)" # e.g check out https://dot.com for more
	userPattern       = '@[^\s]+' # e.g @FagbamigbeK check this out
	alphaPattern      = "[^a-zA-Z0-9]" # e.g I am *10 better!
	sequencePattern   = r"(.)\1\1+"  # e.g Heyyyyyyy, I am back!
	seqReplacePattern = r"\1\1" # e.g Replace Heyyyyyyy with Heyy

	tweet = tweet.lower() #normalizing all text to a lower case
	
	# Replace all URls with 'URL'
	tweet = re.sub(urlPattern,' URL',tweet) #using the substitution method of the regular expression library
	
	# Replace all emojis.
	for emoji in emojis.keys(): #in each of the looped tweet, replace each emojis with their respective meaning
		tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])  # emojis[emoji] helps to get the value of the emoji from the dictionary
		
	# Replace @USERNAME to 'USER'.
	tweet = re.sub(userPattern,' USER', tweet)  #To hide Personal Information, we can replace all usernames with User
	
	# Replace all non alphabets.
	tweet = re.sub(alphaPattern, " ", tweet) # e.g I am *10 better!
	
	# Replace 3 or more consecutive letters by 2 letter.
	tweet = re.sub(sequencePattern, seqReplacePattern, tweet) # e.g Replace Heyyyyyyy with Heyy
	
	tweetwords = ''
	for word in tweet.split():
		if len(word) > 2 and word.isalpha():
			word = wordLemma.lemmatize(word)
			tweetwords += (word + ' ')
		
	return tweetwords

def load_models():
	'''
	Replace '..path/' by the path of the saved models.
	'''

	# Load the vectoriser.
	file = open('models/vectoriser-ngram-(1,2).pickle', 'rb')
	vectoriser = pickle.load(file)
	file.close()

	# Load the LR Model.
	file = open('models/Sentiment-LR.pickle', 'rb')
	LRmodel = pickle.load(file)
	file.close()

	return vectoriser, LRmodel

def predict_pandas(vectoriser, model, cleaned_text):
	textvector = vectoriser.transform([cleaned_text])
	return model.predict_proba(textvector)[0]