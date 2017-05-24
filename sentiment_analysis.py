# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def main():

	data = pd.read_csv("./datasets/finaldataset1.csv", error_bad_lines=False)
	data.columns = ['sentiment', 'content']
	print(data.head())

	print("No. of samples: "), len(data)
	data.dropna() 
	print("Dropped null rows with null values: "), len(data)

	print("Size of dataset:"), data.shape
	print("Columns:"), data.columns.values



	# Pre-processing steps

	import re

	# Hashtags
	hash_regex = re.compile(r"#(\w+)")
	def hash_repl(match):
		return '__HASH_'+match.group(1).upper()

	# Handels
	hndl_regex = re.compile(r"@(\w+)")
	def hndl_repl(match):
		return '__HNDL'#_'+match.group(1).upper()

	# URLs
	url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")

	# Spliting by word boundaries
	word_bound_regex = re.compile(r"\W+")

	# Repeating words like hurrrryyyyyy
	rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE);
	def rpt_repl(match):
			return match.group(1)+match.group(1)

	# Emoticons
	emoticons = \
		[	('__EMOT_SMILEY',	[':-)', ':)', '(:', '(-:', ] )	,\
			('__EMOT_LAUGH',		[':-D', ':D', 'X-D', 'XD', 'xD', ] )	,\
			('__EMOT_LOVE',		['<3', ':\*', ] )	,\
			('__EMOT_WINK',		[';-)', ';)', ';-D', ';D', '(;', '(-;', ] )	,\
			('__EMOT_FROWN',		[':-(', ':(', '(:', '(-:', ] )	,\
			('__EMOT_CRY',		[':,(', ':\'(', ':"(', ':(('] )	,\
		]

	# Punctuations
	punctuations = \
		[	#('',		['.', ] )	,\
			#('',		[',', ] )	,\
			#('',		['\'', '\"', ] )	,\
			('__PUNC_EXCL',		['!', '¡', ] )	,\
			('__PUNC_QUES',		['?', '¿', ] )	,\
			('__PUNC_ELLP',		['...', '…', ] )	,\
			#FIXME : MORE? http://en.wikipedia.org/wiki/Punctuation
		]

	#Printing functions for info
	def print_config(cfg):
		for (x, arr) in cfg:
			print x, '\t',
			for a in arr:
				print a, '\t',
		print ''

	def print_emoticons():
		print_config(emoticons)

	def print_punctuations():
		print_config(punctuations)

	#For emoticon regexes
	def escape_paren(arr):
		return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

	def regex_union(arr):
		return '(' + '|'.join( arr ) + ')'

	emoticons_regex = [ (repl, re.compile(regex_union(escape_paren(regx))) ) \
					for (repl, regx) in emoticons ]

	#For punctuation replacement
	def punctuations_repl(match):
		text = match.group(0)
		repl = []
		for (key, parr) in punctuations :
			for punc in parr :
				if punc in text:
					repl.append(key)
		if( len(repl)>0 ) :
			return ' '+' '.join(repl)+' '
		else :
			return ' '

	def processHashtags( 	text, subject='', query=[]):
		return re.sub( hash_regex, hash_repl, text )

	def processHandles( 	text, subject='', query=[]):
		return re.sub( hndl_regex, hndl_repl, text )

	def processUrls( 		text, subject='', query=[]):
		return re.sub( url_regex, ' __URL ', text )

	def processEmoticons( 	text, subject='', query=[]):
		for (repl, regx) in emoticons_regex :
			text = re.sub(regx, ' '+repl+' ', text)
		return text

	def processPunctuations( text, subject='', query=[]):
		return re.sub( word_bound_regex , punctuations_repl, text )

	def processRepeatings( 	text, subject='', query=[]):
		return re.sub( rpt_regex, rpt_repl, text )

	def processQueryTerm( 	text, subject='', query=[]):
		query_regex = "|".join([ re.escape(q) for q in query])
		return re.sub( query_regex, '__QUER', text, flags=re.IGNORECASE )

	def countHandles(text):
		return len( re.findall( hndl_regex, text) )
	def countHashtags(text):
		return len( re.findall( hash_regex, text) )
	def countUrls(text):
		return len( re.findall( url_regex, text) )
	def countEmoticons(text):
		count = 0
		for (repl, regx) in emoticons_regex :
			count += len( re.findall( regx, text) )
		return count

	#FIXME: preprocessing.preprocess()! wtf! will need to move.
	#FIXME: use process functions inside
	def processAll( 		text, subject='', query=[]):

		if(len(query)>0):
			query_regex = "|".join([ re.escape(q) for q in query])
			text = re.sub( query_regex, '__QUER', text, flags=re.IGNORECASE )

		text = re.sub( hash_regex, hash_repl, text )
		text = re.sub( hndl_regex, hndl_repl, text )
		text = re.sub( url_regex, ' __URL ', text )

		for (repl, regx) in emoticons_regex :
			text = re.sub(regx, ' '+repl+' ', text)


		text = text.replace('\'','')
		# FIXME: Jugad

		text = re.sub( word_bound_regex , punctuations_repl, text )
		text = re.sub( rpt_regex, rpt_repl, text )

		return text
	
	from time import time

	def preprocessing(labeled_tweets):
    		#start = time()
    		procTweets = [ (processAll(s),t) for (t,s) in list_of_labeled_tweets]

    		#end = time()
    		#print "\nExecution time: ", (end - start)
    		return procTweets

	def convertToList(procTweets):
    		lisOfProcTweets = []
    		listOfLabels = []

    		newProcTweets = [list(item) for item in procTweets]
    		#labels = [list(i) for i in newProcTweets]

    		for item in newProcTweets:
        		lisOfProcTweets.append(item[0])
        		listOfLabels.append(item[1])
        
    		return lisOfProcTweets, listOfLabels

	list_of_labeled_tweets = data.values.tolist()

	processed_data = preprocessing(list_of_labeled_tweets)

	print("\nAfter processing: \n")
	print("Training data")

	print processed_data[:5]

	Tweets, Labels = convertToList(processed_data)

	print("Prcessed tweets: "), Tweets[:5]
	print " "
	print("Labels: "), Labels[:5]


	print("Datatype of tweets: {0} and labels: {1} after conversion ".format(type(Tweets), type(Labels )))

	X, y = np.array(Tweets), np.array(Labels)
	len(y)

	## Create dataframe and view 

	df = pd.DataFrame({'tweets':Tweets})
	df['labels'] = Labels
	
	print(df.head())

	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction.text import TfidfTransformer
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.pipeline import FeatureUnion

	## CountVectorizer and TF-IDF Vectors are combined to form feature vectors for the whole dataset

	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.decomposition import TruncatedSVD

	count_vectorizer = CountVectorizer(analyzer="word", binary = False, ngram_range=(1,3), stop_words=None) 
                                    # max_features=10000, \
	tf_transformer = TfidfVectorizer(analyzer="word", ngram_range=(1,3), sublinear_tf=True, use_idf=True) # max_features=10000, 
    
	def findTrainingFeatures(rawTweets):
    		#features = FeatureUnion([("counts", count_vectorizer), ("tfidf", tf_transformer)]).fit_transform(rawTweets)
    		#features = count_vectorizer.fit_transform(df['tweets'])
    		features = tf_transformer.fit_transform(rawTweets)
    
    		return features

	def findTestingFeatures(sentence):
       
    		#features_for_new_input = count_vectorizer.transform(sentence) ## countvectorizer features
    		features_for_new_input = tf_transformer.transform(sentence) ## tf-idf features
    
    		#features_for_new_input = FeatureUnion([("counts", count_vectorizer), ("tfidf", tf_transformer)]).transform(sentence)
    
    		return features_for_new_input

	def predictLabel(testingFeatures):
    		class_name = classifier.predict(testingFeatures) 
    		## classifier.predict_log_proba(combined_features_for_new_input)
    
    		return class_name[0]

	features = findTrainingFeatures(df['tweets'])


	from sklearn.naive_bayes import MultinomialNB, GaussianNB

	classifier = MultinomialNB()
	classifier.fit(features, y)

	print("Enter Input :")
	enter_input = raw_input()

	processedInput = [processAll(enter_input)] ## Pre-process input text
	testingFeatures = findTestingFeatures(processedInput)
	class_name = predictLabel(testingFeatures)
	print("Class predicted: "), class_name

	class_dict = {  '0' : 'anger',  
                	'1' : 'boredom',
                	'2' : 'empty',
                	'3' : 'enthusiasm',
                	'4' : 'fun',
                	'5' : 'happiness',
                	'6' : 'hate',
                	'7' : 'love',
                	'8' : 'neutral',
                	'9' : 'relief',
                	'10' : 'sadness',
                	'11' : 'surprise',
                	'12' : 'worry' }
    
	def ViewClassName(predictedclass):
    		for label, sentiment in class_dict.iteritems():
        		if label == predictedclass:
              			value = sentiment
    		return value

	columns = ['Classifier', 'Parameters optimized', 'k', 'Run time', 'Accuracy']

	results = pd.DataFrame(columns=columns)

	## Classifier : Multinomial Naive Bayes

	from sklearn.cross_validation import cross_val_score
	from sklearn.cross_validation import StratifiedKFold
	from sklearn.grid_search import GridSearchCV

	from time import clock

	start = clock()

	nb_clf_multi = MultinomialNB()

	parameter_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

	cross_validation = StratifiedKFold(y, n_folds=10)

	grid_search = GridSearchCV(nb_clf_multi, param_grid=parameter_grid, cv=cross_validation)

	grid_search.fit(features, y)

	end = clock()
	elapsed = end - start

	print('Best score: {}'.format(grid_search.best_score_))
	print('Best parameters: {}'.format(grid_search.best_params_))

	grid_search.best_estimator_
	
	parameters = []

	for parameter_values in parameter_grid:
    		parameters.append(parameter_values) 

	parameters

	results = results.append({'Classifier': 'Multinomial Naive Bayes', 'Parameters optimized' : parameters, 
                          'k' : cross_validation.n_folds, 'Run time' : elapsed ,
                          'Accuracy' : grid_search.best_score_}, ignore_index=True)
	results

if __name__ == '__main__':
	main()
