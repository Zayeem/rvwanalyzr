from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import models, corpora
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
import string
import random
from nltk.sentiment import SentimentIntensityAnalyzer
import json
import os
import sys
import nltk

#documents, bow, reviews are in the same order.
#   reviews [{id:x, url:x, score:x, text:x, ...}]
#   documents [(review, 'pos'/'neg')] i.e. binary sentiment labeled based on review's score (1-3 = neg, 4,5 = pos)
#   bow [[(word, count)] for lemmatized words of review text]
#   vader_sent_df[(id, score, pos_score, neu_score, neg_score, comp_score)]
#   nb_df [nb_label, id] lable for the review produced by naive_bayes classifier
#   topic_df = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Topic_Keywords', 'id'] of LDA model
class sent_model_builder:

    reviews = []
    all_words = []
    documents = []
    nb_df = pd.DataFrame()
    vader_sent_df = pd.DataFrame()
    topic_df = pd.DataFrame()

    def __init__(self, reviews):
        self.reviews = reviews

    ## Include Title and the review text to analyze the review
    def review_text(self, review):
        if (review):
            return review['title'] + " " + review['text']
        return ""

    ##############VADER sentiment
    def build_vader(self):
        print('\nProcessing VADER sentiment: \n')
        analyser = SentimentIntensityAnalyzer()
        vader_sent = []
        for r in reviews:
            text = self.review_text(r)
            snt = analyser.polarity_scores(text)
            vader_sent.append((r['appid'],r['id'], r['score'], snt.get('neg'), snt.get('neu'), snt.get('pos'), snt.get('compound')))

            #print a few samples randomly for debugging
            if random.randint(1, 200) == 1:
                print('score: {}'.format(r['score']))
                print('AppId: ' + r['appid'])
                print('id: ' + r['id'])
                print('sentiment: ' + str(snt))
                print(' ')

        self.vader_sent_df = pd.DataFrame(vader_sent, columns=['app_id','review_id', 'review_score', 'vader_neg', 'vader_neu', 'vader_pos', 'vader_compound'])
    
    ################# Naive Bayes
    def build_naive_bayes_model(self):
        print('Processing Naive Bayes classification: \n')
        for r in reviews:

            #tokenize review text
            tokens = word_tokenize(self.review_text(r))

            #lower case tokens
            tokens = [w.lower() for w in tokens]

            #remove punctuation
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]

            #filter out non-alphabetic words
            words = [word for word in stripped if word.isalpha()]

            #filter stop words
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if not w in stop_words]

            #Frequency distribution
            for w in words:
                self.all_words.append(w)

            #Frequency distribution
            fdist = FreqDist(self.all_words)
            word_features = list(fdist.keys())[:3000]

            #set the text to processced result for model training later
            r['text'] = ' '.join(words)

            #label reviews with 4, 5 ratings as pos and the rest as neg.
            if(r['score'] == 4 or r['score'] == 5):
                self.documents.append((r, "pos"))
            else:
                self.documents.append((r, "neg"))

        def find_features(text):
            words = word_tokenize(text)
            features = {}
            for w in word_features:
                features[w] = (w in words)

            return features

        featuresets = [(find_features(rvw['text']), sentiment) for (rvw, sentiment) in self.documents]
        random.shuffle(featuresets)

        threshold = 0.8
        training_set = featuresets[:int(threshold*len(featuresets))]    
        testing_set = featuresets[int(threshold*len(featuresets)):]    

        #Prep done. build the model and validate.
        classifier = NaiveBayesClassifier.train(training_set)
  
        print("Naive Bayes classifier accuracy percent:", (nltk_accuracy(classifier, testing_set))*100)
        print("\n")

        classifier.show_most_informative_features(15)

        #build the list of sentiment for each review from the review
        featuresets_to_classify = [find_features(rvw['text']) for (rvw, sentiment) in self.documents]
        labels = classifier.classify_many(featuresets_to_classify)
        self.nb_df = pd.DataFrame(labels, columns = ['nb_label'])
        self.nb_df['review_id'] = [rvw['id'] for rvw, sentiment in self.documents]
        self.nb_df['appid'] = [rvw['appid'] for rvw, sentiment in self.documents]

    ###########################topic modeling
    def build_topic_model(self):
        print('\nProcessing Topic modeling: \n')
        # Lemmatization rather than stemming to preserve the root form.
        prepped = []
        for r in reviews:
            #lemmmatization
            tokens = word_tokenize(self.review_text(r))

            lem = WordNetLemmatizer()
            lemmatized = [lem.lemmatize(w) for w in tokens]

            prepped.append(lemmatized)

        #create gensim dictionary of word<->id mapping
        w2id = corpora.Dictionary(prepped)

        #create term frequence list (word, count) - Bag of words representation
        bow = [w2id.doc2bow(text) for text in prepped]

        #LDA for the review
        num_topics = 10
        ldamodel = models.ldamodel.LdaModel(bow, 
                num_topics=num_topics, id2word=w2id, passes=25)
        
        print("Top 2 topic stats:\n")
        print(ldamodel.show_topics(num_topics=2, num_words=5))
        print("\nSample topics:\n")

        #Using the model built, build topic_df with columns ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords'].
        def format_topics_sentences(ldamodel=ldamodel, corpus=bow):

            sent_topics_df = pd.DataFrame()
            
            # Get main topic in each document
            for row in ldamodel[corpus]:
                row = sorted(row, key=lambda x: (x[1]), reverse=True)
                # Get the Dominant topic, Perc Contribution and Keywords for each document
                for j, (topic_num, prop_topic) in enumerate(row):
                    if j == 0:  # => dominant topic
                        wp = ldamodel.show_topic(topic_num)
                        topic_keywords = ", ".join([word for word, prop in wp])
                        
                        #print a few samples for debugging
                        if random.randint(1, 200) == 1:
                            print(topic_keywords)

                        sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                    else:
                        break

            sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

            return(sent_topics_df)


        df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel, corpus=bow)
        print("\n Perplexity:" , ldamodel.log_perplexity(bow))
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=ldamodel, dictionary=w2id, texts=prepped,
                                              coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        self.topic_df = df_topic_sents_keywords.reset_index()
        self.topic_df.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Topic_Keywords']
        self.topic_df['review_id'] = [rvw['id'] for rvw, sentiment in self.documents]
        self.topic_df['appid'] = [rvw['appid'] for rvw, sentiment in self.documents]


#main runs review data through VADER sentiment analyzer, naive bayes classifier, LDA topic modeler and
#creates csv files containing the results for each in the current directory.
#TODOs:
#-Load reviews in chronological order to analyze chronological trend.
#-Anaylyze the saved csv further.
#-Save the trained model to provide a function to load the trained model and anaylyze a new review. 
#   i.e. a function to propose a rate given review text. (This requires adding multi_classfication function)
#-Possibly a graphical representation of the combined csv. i.e. topic keywords displayed along with 
#   the sentiment(vader, nb) of each review or a new review.
if __name__=='__main__':

    # Download the prerequisite corpora
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    # Download tokenizer
    nltk.download('punkt')

    if len(sys.argv) != 2:
        print("Usage: python sent_model_builder.py path_to_data_dir")
        print("Setting path_to_data_dir to app-store-scripts/data/apps by default\n")
        dataDir = "app-store-scripts/data/apps"
    else:
        dataDir = sys.argv[1]
        
    reviews = []

    #Load reviews from all folders with files named reviews-0.json ... reviews-9.json.
    #nb model accuracy gets to ~90% if at least 2 review folders are provided. ~70% from a single review folder.
    for dir in sorted(os.listdir(dataDir)):
        if not dir.startswith('.') and os.path.isdir(os.path.join(dataDir, dir)):
            for filename in sorted(os.listdir(os.path.join(dataDir, dir))):
                if not filename.startswith('.'):
                    filepath = os.path.join(dataDir, dir, filename) 
                    with open(filepath) as json_file:
                        json_vals = json.load(json_file)
                        for j in json_vals:
                            j["appid"] = dir
                        reviews.extend(json_vals)

    #build models
    model_builder = sent_model_builder(reviews)
    model_builder.build_vader()
    model_builder.build_naive_bayes_model()
    model_builder.build_topic_model()

    #save the resulting data as csv in current directory
    if not os.path.isdir('output'):
        os.mkdir('output')
        
    model_builder.topic_df.to_csv('output/topic.csv') #Top topic of each review
    model_builder.nb_df.to_csv('output/nb.csv') #sentiment label of each review
    model_builder.vader_sent_df.to_csv('output/vader.csv') #vader sentiment score of each review

    exit(0)