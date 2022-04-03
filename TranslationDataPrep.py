import re
import numpy
from pickle import dump
from unicodedata import normalize
from numpy import array
from pickle import load
from numpy.random import rand
from numpy.random import shuffle
import string
import codecs

SPECIAL_UNICODE_HANDLING=False

class TranslationDataPrep:

    def __init__(self, pairsFilePath, allPickle, cleanPickle, trainPickle, testPickle, maxNumOfSentences, trainTestSplit):
        # local data
        self.allPickle=allPickle
        self.trainPickle = trainPickle
        self.testPickle = testPickle
        self.cleanPickle = cleanPickle
        self.maxNumOfSentences = maxNumOfSentences
        
        # load dataset
        filename = pairsFilePath  
        doc = self.load_doc(filename)
        # split into english-<other language> pairs
        pairs = self.to_pairs(doc)
        # clean sentences
        clean_pairs = self.clean_pairs(pairs)
        # spot check
        for i in range(10):
            print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))
        # save clean pairs to file
        self.save_clean_data(clean_pairs, self.allPickle ) 
         
        # load dataset
        self.raw_dataset = self.load_clean_sentences(self.allPickle)
 
        # reduce dataset size
        self.n_sentences = maxNumOfSentences # 10000
        self.dataset = self.raw_dataset[:maxNumOfSentences, :]
        # random shuffle
        shuffle(self.dataset)
        # split into train/test
        boundary = int(trainTestSplit*len(self.dataset))
        self.train, self.test = self.dataset[:boundary], self.dataset[boundary:]
        # save
        self.save_clean_data(self.dataset, self.cleanPickle)
        self.save_clean_data(self.train, self.trainPickle)
        self.save_clean_data(self.test, self.testPickle)

    
    def to_pairs(self, doc):
        lines = doc.strip().split('\n')
        pairs = [line.split('\t') for line in  lines]
        return pairs
    
    def clean_pairs(self, lines):
        cleaned = list()
        # prepare regex for char filtering
        re_print = re.compile('[^%s]' % re.escape(string.printable))
        # prepare translation table for removing punctuation
        table = str.maketrans('', '', string.punctuation)
        for pair in lines:
            clean_pair = list()
            for line in pair:
                # normalize unicode characters
                line = normalize('NFD', line).encode('ascii', 'ignore')
                line = line.decode('UTF-8')
                # tokenize on white space
                line = line.split()
                # convert to lowercase
                line = [word.lower() for word in line]
                # remove punctuation from each token
                line = [word.translate(table) for word in line]
                # remove non-printable chars form each token
                line = [re_print.sub('', w) for w in line]
                # remove tokens with numbers in them
                line = [word for word in line if word.isalpha()]
                # store as string
                clean_pair.append(' '.join(line))
            cleaned.append(clean_pair)
        return array(cleaned)
    
    def printuni(self, ustr):
        print(ustr.encode('raw_unicode_escape'))
        
    def saveuni(self, ustr, filename='output.txt'):
        file(filename,'wb').write(codecs.BOM_UTF8 + ustr.encode('utf8'))
     
    # load doc into memory
    def load_doc(self, filename):
        # open the file as read only
        file = open(filename, mode='rt', encoding='utf-8')
        # read all text
        text = file.read()
        #print("text=",text)
        # close the file
        file.close()
        return text
     
    # split a loaded document into sentences
    def to_pairs(self, doc):
        lines = doc.strip().split('\n')
        pairs = [line.split('\t') for line in  lines]
        return pairs
     
    # clean a list of lines
    def clean_pairs(self, lines):
        cleaned = list()
        # prepare regex for char filtering
        re_print = re.compile('[^%s]' % re.escape(string.printable))
        # prepare translation table for removing punctuation
        table = str.maketrans('', '', string.punctuation)
        for pair in lines:
            clean_pair = list()
            for line in pair:
                #print("line in pair=",line)
                # normalize unicode characters
                if SPECIAL_UNICODE_HANDLING:
                    line = normalize('NFD', line).encode('ascii', 'ignore')
                    line = line.decode('UTF-8')
                # tokenize on white space
                line = line.split()
                #print("splitted line=",line)
                # convert to lowercase
                line = [word.lower() for word in line]
                #print("tokenized line=",line)
                # remove punctuation from each token
                line = [word.translate(table) for word in line]
                #print("-punct line=",line)
                # remove non-printable chars form each token
                #line = [re_print.sub('', w) for w in line]
                #print("printable line=",line)
                # remove tokens with numbers in them
                line = [word for word in line if word.isalpha()]
                #print("line withut numbers=",line)
                # store as string
                clean_pair.append(' '.join(line))
            cleaned.append(clean_pair)
        return array(cleaned)
     
    # save a list of clean sentences to file
    def save_clean_data(self, sentences, filename):
        dump(sentences, open(filename, 'wb'))
        print('Saved: %s' % filename)
    
    
    # load a clean dataset
    def load_clean_sentences(self, filename):
        return load(open(filename, 'rb'))
    
    # save a list of clean sentences to file
    def save_clean_data(self, sentences, filename):
        dump(sentences, open(filename, 'wb'))
        print('Saved: %s' % filename)


