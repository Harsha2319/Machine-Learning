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
import numpy as np
import pandas as pd
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
    
"""
input : list of file names and path to the directory
output: list of all words in all files in the directory, list of unique words
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
    return train_words, train_vocab

#Generating dataset with each record containing frequency of unique words in trainset for each file
def generate_dataset(vocab, path, file_names, target):
    j = 0
    d = []
    for i in file_names:
        j = j + 1
        with open(path+i, 'r', errors='ignore') as file:
            train_f = file.read()
            file.close()
        train_words_f = re.split(' |\n',train_f)
        train_freq_f = Counter(list(train_words_f))
        record_val = []
        for i in vocab:
            if (i not in train_freq_f.keys()):
                record_val.append(0)
            else:
                record_val.append(train_freq_f[i])
        record_val.append(target)
        d.append(record_val)
    return d

#Converts list of list into dataframe
def to_dataframe(List, col):
    array = np.array(List)
    dic = {}
    for i,j in zip(range(len(List[0])),col):
        dic[j] = list(array[:,i])
    df = pd.DataFrame(dic, columns=col)
    return df

def main():
    print("Collecting vocab from Train Spam")
    train_words_spam, train_vocab_spam = text_clean(path1, trains)
    print("No of Train words - spam : ",len(train_words_spam))
    print("No of unique Train words - spam : ",len(train_vocab_spam))
    
    print("Collecting vocab from Train ham")
    train_words_ham, train_vocab_ham = text_clean(path2, trainh)
    print("No of Train words - ham : ",len(train_words_ham))
    print("No of unique Train words - ham : ",len(train_vocab_ham))
    train_vocab = train_vocab_spam + train_vocab_ham
    col = train_vocab + ["class"]
    
    print("Generating train dataset")
    d1 = generate_dataset(train_vocab, path1, trains, 0)
    d2 = generate_dataset(train_vocab, path2, trainh, 1)
    train_list = d1 + d2
    train_df = to_dataframe(train_list, col)
    
    print("Generating test dataset")
    d3 = generate_dataset(train_vocab, path3, tests, 0)
    d4 = generate_dataset(train_vocab, path4, testh, 1)
    test_list = d3 + d4
    test_df = to_dataframe(test_list, col)
    
    #Saving train dataset to 'train.csv'
    train_df.to_csv('train.csv', index=False)
    #saving test dataset to 'test.csv'
    test_df.to_csv('test.csv', index=False)
    
    print('---------------------------------------------------------------------------------------------------------------------')
    
    """ ***************************************************************************************************************************
    Generating dataframes without stop words 
    
    **************************************************************************************************************************** """
    
    new_train_words_spam = [i for i in train_words_spam if i not in stop_words]
    new_train_vocab_spam = (list(set(new_train_words_spam)))
    print("No of Train words - spam - No stop words : ",len(new_train_words_spam))
    print("No of unique Train words - spam - No stop words : ",len(new_train_vocab_spam))
    new_train_words_ham = [i for i in train_words_ham if i not in stop_words]
    new_train_vocab_ham = (list(set(new_train_words_ham)))
    print("No of Train words - ham - No stop words : ",len(train_words_ham))
    print("No of unique Train words - ham - No stop words : ",len(train_vocab_ham))
    new_train_vocab = new_train_vocab_spam + new_train_vocab_ham
    col1 = new_train_vocab + ["class"]
    
    print("Generating train dataset - No stop words")
    d1 = generate_dataset(new_train_vocab, path1, trains, 0)
    d2 = generate_dataset(new_train_vocab, path2, trainh, 1)
    train_list = d1 + d2
    new_train_df = to_dataframe(train_list, col1)
    
    print("Generating test dataset - No stop words")
    d3 = generate_dataset(new_train_vocab, path3, tests, 0)
    d4 = generate_dataset(new_train_vocab, path4, testh, 1)
    test_list = d3 + d4
    new_test_df = to_dataframe(test_list, col1)
    
    new_train_df.to_csv('new_train.csv', index=False)
    new_test_df.to_csv('new_test.csv', index=False)
    
main()

    