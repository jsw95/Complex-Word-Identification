from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from nltk.corpus import cmudict
from nltk.corpus import wordnet as wn
from utils.dataset import Dataset
import nltk
from collections import Counter 


### BASELINE CLASS USED TO IDENTIFY COMPLEX WORDS ###

class System(object):

    # initialised with a choice of language, model and features
    def __init__(self, language, modelName, featureSet):

        
        self.language = language
        self.modelName = modelName
        self.featureSet = featureSet


        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3

        else:  # spanish
            self.avg_word_length = 6.2


        if modelName == 'RandomForest':
            self.model = RandomForestClassifier()

        elif modelName == 'NeuralNetwork':
            self.model = MLPClassifier()

        elif modelName == 'SVC':
            self.model = SVC()        

        elif modelName == 'LogisticRegression':
            self.model = LogisticRegression()

        # Building the frequency counts from the data
        self.word_count, self.uni_count, self.bi_count, self.tri_count = self.word_freq(language)
       
        # The possible POS tags, needed to ensure equal vector lengths for each sentence
        pos_tags_nltk = ['cc', 'cd','dt','in','jj','jjr','jjs','nn','nns',
        'nnp','nnps','pdt','pos','prp','prp$','rb','rbr','rbs','rp',
        'sym','vb','vbd','vbg','vbn','vbp','vbz','wdt','wp','wp$','wrb']

        self.vec = CountVectorizer(vocabulary = pos_tags_nltk)
        
        # NLTK dict of syllables in words
        self.syl_dict = cmudict.dict()

##########################################################################################
    
    ### PREPROCESSING FUNTIONS FOR FEATURE EXTRACTION


    # maximum number of syllables in a phrase
    def avg_syl(self, sent):   
        sent_syls = []
        for word in sent.split(" "):
            # checking wether word is in cmudict
            if word.lower() in self.syl_dict:
                # generates a list of number of syllables for each word
                sent_syls.extend([len(list(y for y in x if y[-1].isdigit())) 
                    for x in self.syl_dict[word.lower()]])

            if len(sent_syls) != 0: return sum(sent_syls)/len(sent_syls) #avg senses
            else: return 0

    # reads the trainset to build word, letter bigram and trigram frequency dicts
    def word_freq(self, language, unigram = True, bigram = True, trigram = True):
        data = Dataset(language)
        word_count = Counter()
        uni_count = Counter()
        bi_count = Counter()
        tri_count = Counter()

        text = []
        for line in data.trainset:
            if line['sentence'] not in text:
                text.append(line['sentence'])

        # Building unigram wordcounts
        words = ' '.join(text)
        for word in words.split(" "):
            word_count[word] += 1

            # Building letter Unigram counts
            if unigram == True:
                for i in range(len(word) - 1):
                    uni_count[word[i]] += 1

            # Building letter Bigram counts
            if bigram == True:
                for i in range(len(word) - 1):
                    bi_count[word[i:i+2]] += 1

            # Building letter Trigram counts
            if trigram == True:
                for i in range(len(word) - 2):
                    tri_count[word[i:i+3]] += 1

        return word_count, uni_count, bi_count, tri_count

    # returns how frequent a specific word, letter bi/trigram is
    def letter_count(self, word, unigram = False, bigram = False, trigram = False):
        counts = []

        if unigram == True:
            for token in word.split(" "):
                for i in range(len(token)):
                    counts.append(self.uni_count[token[i:i+1]])

        if bigram == True:
            for token in word.split(" "):
                for i in range(len(token) - 1):
                    counts.append(self.bi_count[token[i:i+2]])
            
        if trigram == True:
            for token in word.split(" "):
                for i in range(len(token) - 2):
                    counts.append(self.tri_count[token[i:i+3]])

        else: pass

        if len(counts) == 0: avg = 0 
        else: avg = sum(counts)/len(counts)  #avg 0.77 min 0.75

        return avg

    # return the average number of senses for each word
    def avg_sense(self, word):
        num_sense = [len(wn.synsets(token)) for token in word.split(" ") 
        if len(wn.synsets(token)) != 0] # List of senses for each word
      
        if len(num_sense) > 0: avg_sense = sum(num_sense)/len(num_sense)
        else: avg_sense = 20

        return avg_sense

    # returns a one hot vector of pos tags
    def pos_counts(self, word):
        pos = [' '.join(i[1] for i in nltk.pos_tag(nltk.word_tokenize(word)))]
        X = self.vec.fit_transform(pos)
        pos_counts = X.toarray()[0]
        return pos_counts

    def capital_letters(self, word):
        caps = [int(token[0].isupper()) for token in word.split(" ") if token]
        return sum(caps)

############################################################################################

    #### EXTRACTING SET OF FEATURES FROM EACH WORD ###

    def extract_features(self, word):

        feats = []

        # baseline features
        if 'baseline' in self.featureSet:
            len_chars = len(word) / self.avg_word_length # relative word length
            len_tokens = len(word.split(' ')) # number of tokens in 'word'(may be sentence)
            feats.extend([len_chars, len_tokens])

        # word starts with a capital letter
        if 'cap_feat' in self.featureSet:
            feats.append(self.capital_letters(word))

        # word unigram frequency
        if 'freq_feat' in self.featureSet:
            feats.append(self.word_count[word])


        # letter unigrams frequency
        if 'uni_feat' in self.featureSet:
            feats.append(self.letter_count(word, unigram = True))

        # letter bigrams frequency
        if 'bi_feat' in self.featureSet:
            feats.append(self.letter_count(word, bigram = True))
            
        # letter trigrams frequency                   
        if 'tri_feat' in self.featureSet:
            feats.append(self.letter_count(word, trigram = True))
            
                             
        # max number of syllables for each word
        if 'syl_feat' in self.featureSet:
            feats.append(self.avg_syl(word))


        # average number of other senses for the words
        if 'sense_feat' in self.featureSet:
            feats.append(self.avg_sense(word))

        # Word POS tags one hot encoded 
        if 'pos_feat' in self.featureSet:
            feats.extend(self.pos_counts(word))

        return feats

#####################################################################################


    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:

            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])
      
        self.model.fit(X, y)

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word']))
                
         
        return self.model.predict(X)



if __name__ == '__main__':
    pass







