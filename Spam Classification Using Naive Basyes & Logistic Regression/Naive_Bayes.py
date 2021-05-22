stop_words = ['a','about','above','after','again','against','all','am','an','and','any','are','aren\'t','as','at','be','because',
'been','before','being','below','between','both','but','by','can\'t','cannot','could','couldn\'t','did','didn\'t','do','does',
'doesn\'t','doing','don\'t','down','during','each','few','for','from','further','had','hadn\'t','has','hasn\'t','have','haven\'t',
'having','he','he\'d','he\'ll','he\'s','her','here','here\'s','hers','herself','him','himself','his','how','how\'s','i',
'i\'d','i\'ll','i\'m','i\'ve','if','in','into','is','isn\'t','it','it\'s','its','itself','let\'s','me','more','most','mustn\'t',
'my','myself','no','nor','not','of','off','on','once','only','or','other','ought','our','ours','ourselves','out','over','own',
'same','shan\'t','she','she\'d','she\'ll','she\'s','should','shouldn\'t','so','some','such','than','that','that\'s','the','their',
'theirs','them','themselves','then','there','there\'s','these','they','they\'d','they\'ll','they\'re','they\'ve','this','those',
'through','to','too','under','until','up','very','was','wasn\'t','we','we\'d','we\'ll','we\'re','we\'ve','were','weren\'t','what',
'what\'s','when','when\'s','where','where\'s','which','while','who','who\'s','whom','why','why\'s','with','won\'t','would',
'wouldn\'t','you','you\'d','you\'ll','you\'re','you\'ve','your','yours','yourself','yourselves']

import os
import re
import math
from collections import Counter
path1 = 'train\\spam\\'
path2 = 'train\\ham\\'
path3 = 'test\\spam\\'
path4 = 'test\\ham\\'
trains = os.listdir(path1)
trainh = os.listdir(path2)
tests = os.listdir(path3)
testh = os.listdir(path4)
print("Number of Spam files in training set : ",len(trains))
print("Number of ham files in training set : ",len(trainh))
print("Number of Spam files in testing set : ",len(tests))
print("Number of ham files in testing set : ",len(testh))

train_words_spam = []
train_vocab_spam = []
train_freq_spam = {}
train_words_ham = []
train_vocab_ham = []
train_freq_ham = {}
p_spam = len(trains)
p_ham = len(trainh) 

"""
input : list of file names and path to the directory
output: list of all words in all files in the directory, list of unique words, frequency of all unique words
"""
def text_clean(path, file_names):
    train_vocab = []
    train_words = []
    for i in file_names:
        with open(path+i, 'r', errors='ignore') as file:
            train_f = file.read()
            file.close()
        train_words_f = re.split(' |\n',train_f)
        train_words = train_words + train_words_f

    train_vocab = (list(set(train_words)))
    train_freq = Counter(list(train_words))
    return train_words, train_vocab, train_freq
        
#classifies each record of the test set
def classify(path, file_names, train_words_spam, train_vocab_spam, train_freq_spam, train_words_ham, train_vocab_ham, train_freq_ham):
    a = 0
    b = 0
    prediction = []
    for i in file_names:
        with open(path+i, 'r', errors='ignore') as file:
            test_f = file.read()
            file.close()
        test_f_words = re.split(' |\n',test_f)
        test_vocab = (list(set(test_f_words)))
            
        p_test_spam = 1
        p_test_ham = 1
    
        for i in test_vocab:
            if (not(i in train_freq_spam)):
                train_freq_spam[i] = 0
            if (not(i in train_freq_ham)):
                train_freq_ham[i] = 0
            p_i_spam =(train_freq_spam[i]+1)/((len(train_words_spam))+(len(train_vocab_spam)))
            p_i_ham =(train_freq_ham[i]+1)/((len(train_words_ham))+(len(train_vocab_ham)))
            p_test_spam = p_test_spam + math.log2(p_i_spam)
            p_test_ham = p_test_ham + math.log2(p_i_ham)
        p_test_spam = p_test_spam + math.log2(p_spam)
        p_test_ham = p_test_ham + math.log2(p_ham)
        if (p_test_spam > p_test_ham):
            prediction.append("spam")
            a = a + 1
        else:
            prediction.append("ham")
            b = b + 1
    return a, b

def accuracy(TP, TN, FP, FN):
    print(TP, TN, FP, FN)
    return((TP + FN )/( TP + TN + FP + FN))*100
        
def main():
    print("Training Spam")
    train_words_spam, train_vocab_spam, train_freq_spam = text_clean(path1, trains)
    print("No of Train words - spam : ",len(train_words_spam))
    print("No of unique Train words - spam : ",len(train_vocab_spam))
    
    print("Training ham")
    train_words_ham, train_vocab_ham, train_freq_ham = text_clean(path2, trainh)
    print("No of Train words - ham : ",len(train_words_ham))
    print("No of unique Train words - ham : ",len(train_vocab_ham))
    
    print("Test Spam")
    spam_t, spam_f = classify(path3, tests, train_words_spam, train_vocab_spam, train_freq_spam, train_words_ham, train_vocab_ham, train_freq_ham)
    
    print("Test Ham")
    ham_f, ham_t = classify(path4, testh, train_words_spam, train_vocab_spam, train_freq_spam, train_words_ham, train_vocab_ham, train_freq_ham)
    
    acc = accuracy(spam_t, spam_f, ham_f, ham_t)
    print("Accuracy : ", acc,"%")
    
    print("-----------------------------------------------------------------------------------------------------------------------------------")
    
    new_train_words_spam = [i for i in train_words_spam if i not in stop_words]
    new_train_vocab_spam = (list(set(new_train_words_spam)))
    new_train_freq_spam = Counter(list(new_train_words_spam))
    print("No of Train words - spam - No stop words : ",len(new_train_words_spam))
    print("No of unique Train words - spam - No stop words : ",len(new_train_vocab_spam))
    
    new_train_words_ham = [i for i in train_words_ham if i not in stop_words]
    new_train_vocab_ham = (list(set(new_train_words_ham)))
    new_train_freq_ham = Counter(list(new_train_words_ham))
    print("No of Train words - ham - No stop words : ",len(new_train_words_ham))
    print("No of unique Train words - ham - No stop words : ",len(new_train_vocab_ham))
    
    print("Test Spam - No Stop Words")
    new_spam_t, new_spam_f = classify(path3, tests, new_train_words_spam, new_train_vocab_spam, new_train_freq_spam, new_train_words_ham, new_train_vocab_ham, new_train_freq_ham)
    
    print("Test Ham - No Stop Words")
    new_ham_f, new_ham_t = classify(path4, testh, new_train_words_spam, new_train_vocab_spam, new_train_freq_spam, new_train_words_ham, new_train_vocab_ham, new_train_freq_ham)
    
    new_acc = accuracy(new_spam_t, new_spam_f, new_ham_f, new_ham_t)
    print("Accuracy - No Stop Words : ", new_acc,"%")
    
main()
    