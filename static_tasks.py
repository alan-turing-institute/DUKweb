#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pickle, re, gensim
from sklearn.metrics.pairwise import cosine_similarity 

'''
a) Download the word analogy dataset by Mikolov et al. 2013 
and place in:  data/resources/word_analogy.txt
(available at http://download.tensorflow.org/data/questions-words.txt)

b) Download the word similarity and relatedness datasets from 
and place them in data/resources/ (wordsim_relatedness_goldstandard.txt and
wordsim_relatedness_goldstandard.txt). 


c) Download word2vec-google-news-300 from gensim. 
Place in data/resources/word2vec-google-news-300

d) Download the 100-dim GloVe vectors from https://nlp.stanford.edu/projects/glove/
Place them in data/resources/glove.twitter.27B.100d.txt

e) The TRI and SGNS vectors in our dataset should be placed in the 
data/resources/TRI/ and data/resources/SGNS/ folders, respectively
'''



'''Seven methods to load the following:
        (i) each of the 4 models 
        (ii) their joint vocabulary
        (iii) the datasets for the 3 tasks (analogy/similarity/relaterness)
'''
 
def read_glove(folder='data/glove/', tof='twitter'):
    '''(i) Baseline Reader: GloVe
        > Requires the filepath to the GloVe embeddings (twitter, 100d) 
        > Returns: a dictionary of: {word->embedding}
    '''
    vocab = get_vocabulary()
    infile = folder+'glove.twitter.27B.100d.txt'
    with open(infile, 'r') as f:
        lines = f.readlines()
    f.close()
    data = dict()
    for line in lines:
        fields = line.split()
        word = str(fields[0])
        if word in vocab:
            data[word] = np.array([np.float(fields[i]) for i in range(1,len(fields))])
    return data


def read_word2vec(folder='data/word2vec/'):
    '''(i) Baseline Reader: word2vec on google news
        > Requires the filepath to the word2vec embeddings (300d) 
        > Returns: a dictionary of: {word->embedding}
    '''
    vocab = get_vocabulary()
    infile = folder+'word2vec-google-news-300'
    raw = gensim.models.KeyedVectors.load_word2vec_format(infile, binary=True)
    model = dict()
    for word in vocab:
        model[word] = raw[word]
    return model


def read_TRI(year, folder='data/TRI/'):
    '''(i) Reader: TRI
        > Requires the year to load (2000-2012) and the local path to the TRI vectors
        > Returns: a dictionary in the form of {word->vector}
    '''    
    vocab = get_vocabulary()
    with open(folder+'D-'+str(year)+'_merge_occ_tri.text', 'r') as infile:
        lines = infile.readlines()
    infile.close()
    model= dict()
    for line in lines:
        fields = line.split()
        word = str(fields[0]).strip()
        if word in vocab:
            model[word] = np.array([np.float(fields[i]) for i in range(1, len(fields))])
    return model


def read_sgns(year, folder='data/SGNS/'):
    '''(i) Reader: SGNS
        > Requires the year to load (int) and the local path to the SGNS vectors
        > Returns: a dictionary in the form of {word->vector}
    '''    
    vocab = get_vocabulary()
    infile = folder+str(year)+'.csv'
    with open(infile, 'r') as f:
        lines = f.readlines()
    model = dict()
    for line in lines:
        fields = line.split(',')
        word = fields[0]
        if len(fields)!=101:
            for i in range(1, len(fields)-100):
                word+=','+str(fields[i])
        if word in vocab:
            model[word] = np.array([np.float(fields[i]) for i in range(1, len(fields))])
            if len(model[word])!=100:
                print('Error:', fields[0])
    return model


def get_vocabulary(folder='data/'):
    '''(ii) The joint vocabulary reader'''
    return pickle.load(open(folder+'joint_vocabulary.p', 'rb'))



def get_word_analogy_data(folder='data/resources/'):
    '''(iii) Reads the data used for the word analogy task'''
    infile = folder + 'word_analogy.txt'
    with open(infile, 'r') as f:
        lines = f.readlines()
    f.close()
    
    data = dict()
    for line in lines[1:]:
        if line.startswith(':'):
            title = line.split(' ')[1]
            title = title[0:len(title)-1]
            if title in ['capital-common-countries', 'capital-world', 'city-in-state']:
                title = 'geography'
            elif title[0:4]=='gram':
                title = 'grammar'
        else:
            try:
                preval = data[title]
            except KeyError:
                preval = []
            vals = line.split(' ')
            for i in range(len(vals)):
                vals[i] = re.sub('[^0-9a-zA-Z]+', '', vals[i])
            preval.append(vals)
            data[title] = preval
    return data


def read_wordsim(folder='data/resources/', tof='similarity'):
    '''(iii) Reads the data used for the word relatedness/similarity tasks'''
    infile = folder+'wordsim_'+tof+'_goldstandard.txt'
    with open(infile, 'r') as f:
        lines = f.readlines()
    f.close()
    
    pairs, scores = [], []
    for line in lines:
        fields = line.split()
        if fields[0]!=fields[1]:
            pairs.append([fields[0].lower(), fields[1].lower()])
            scores.append(float(fields[2]))
    return pairs, scores







'''The four methods to be called so that we run everything (one per model)'''


'''The "main" method for TRI'''
def get_all_results_TRI():
    results_analogy, results_relate, results_similar = dict(), dict(), dict()
    analogy = get_word_analogy_data() 
    similar, similar_scores = read_wordsim(tof='similarity')
    relate, relate_scores = read_wordsim(tof='relatedness')
    for modelname in range(2000,2013):
        print(modelname)
        model = read_TRI(modelname)
        results_analogy[modelname] = evaluate_word_analogy(model, analogy, 'TRI')
        results_similar[modelname] = evaluate_wordsim(model, similar, similar_scores, 'TRI')
        results_relate[modelname] = evaluate_wordsim(model, relate, relate_scores, 'TRI')
    return results_analogy, results_similar, results_relate

'''The "main" method for SGNS'''
def get_all_results_SGNS():
    results_analogy, results_relate, results_similar = dict(), dict(), dict()
    analogy = get_word_analogy_data() 
    similar, similar_scores = read_wordsim(tof='similarity')
    relate, relate_scores = read_wordsim(tof='relatedness')
    for modelname in range(2000,2013):
        print(modelname)
        model = read_sgns(modelname)
        results_analogy[modelname] = evaluate_word_analogy(model, analogy)
        results_similar[modelname] = evaluate_wordsim(model, similar, similar_scores)
        results_relate[modelname] = evaluate_wordsim(model, relate, relate_scores)
    return results_analogy, results_similar, results_relate

'''Baseline: pre-trained word2vec'''
def get_all_results_word2vec():
    model = read_word2vec()
    results_analogy, results_relate, results_similar = dict(), dict(), dict()
    analogy = get_word_analogy_data() 
    similar, similar_scores = read_wordsim(tof='similarity')
    relate, relate_scores = read_wordsim(tof='relatedness')
    for modelname in range(2000,2013): #this is redundant
        print(modelname)
        results_analogy[modelname] = evaluate_word_analogy(model, analogy)
        results_similar[modelname] = evaluate_wordsim(model, similar, similar_scores)
        results_relate[modelname] = evaluate_wordsim(model, relate, relate_scores)
    return results_analogy, results_similar, results_relate

'''The "main" method for GloVe models'''
def get_all_results_glove():
    model = read_glove()
    results_analogy, results_relate, results_similar = dict(), dict(), dict()
    analogy = get_word_analogy_data() 
    similar, similar_scores = read_wordsim(tof='similarity')
    relate, relate_scores = read_wordsim(tof='relatedness')
    for modelname in range(2000,2013):
        print(modelname)
        results_analogy[modelname] = evaluate_word_analogy(model, analogy)
        results_similar[modelname] = evaluate_wordsim(model, similar, similar_scores)
        results_relate[modelname] = evaluate_wordsim(model, relate, relate_scores)
    return results_analogy, results_similar, results_relate






            


'''The two functions that are used for performing the task'''
def evaluate_word_analogy(model, dataset, modelname='word2vec'):
    '''Word analogy task'''
    all_scores = dict()
    vocab = model.keys()
        
    for title in dataset.keys(): #i.e., [currency, grammar, geography, family]
        data = dataset[title]    
        scores = []
        for pair in data: #pair is actually a list of 4 items
            flag = True #first check if all words are in the joint vocabulary
            for val in pair: #val is a token
                val = val.lower()
                if val not in vocab:
                    flag = False
            if flag==True: #proceed if all words are present in the vocabulary
                val1, val2, val3, val4 = model[pair[0].lower()], model[pair[1].lower()], model[pair[2].lower()], model[pair[3].lower()]
                if modelname=='word2vec':
                    vall = val1-val2+val4 
                    dist3 = cosine_similarity(val3.reshape(1,-1), vall.reshape(1,-1))[0][0]
                else:
                    vall = orthogonalization(val1, val2, val4)
                    dist3 = cosine_similarity(val3.reshape(1,-1), vall.reshape(1,-1))[0][0]
                scores.append(dist3)
        all_scores[title] = scores
    return all_scores

        
def evaluate_wordsim(model, data, scores, modelname='word2vec'):
    '''Word relatedness and similarity tasks'''
    vocab = model.keys()
        
    results = []
    for i in range(len(data)):
        word1, word2 = data[i][0], data[i][1]
        score = scores[i]
        if (word1 in vocab) & (word2 in vocab):
            val1, val2 = model[word1], model[word2]
            dist = cosine_similarity(val1.reshape(1,-1), val2.reshape(1,-1))[0][0]
            results.append([score, dist])
    return results

 
def orthogonalization(val1, val2, val4):
    '''Helper function used for the TRI vectors'''
    norm1 = val1 / np.linalg.norm(val1)
    norm2 = val2 / np.linalg.norm(val2)
    norm4 = val4 / np.linalg.norm(val4)
    s = norm2 + norm4
    norms = s / np.linalg.norm(s)
    c = norm1.dot(norms)
    o = norm1 - (c * norms)
    normo = o / np.linalg.norm(o)
    return normo