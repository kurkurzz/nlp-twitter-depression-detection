import pandas as pd

from processing import load_models, predict_pandas
vectoriser, LRmodel = load_models()

df = pd.DataFrame([{
		'text': 'i love you'
	},{
		'text': 'i hate you'
	}])

print(df['text'].apply(lambda x: predict_pandas(vectoriser, LRmodel, x)))