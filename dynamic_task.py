#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics.pairwise import cosine_distances as cos_dist
import warnings
warnings.filterwarnings('ignore') 


def run_all_sgns_models(pct_diachronic=0.001, pct_anchors=0.01): 
    '''Call this function to run the three models. The results will be saved in
    the "dynamic_results/" folder. The pct_diachronic/pct_anchors parameters
    define the % of vocab_size to use as diachronic ahcors/anchors words.'''
    words, labels, vectors = read_data() #three lists: words, labels (change/static) and vector representations
    distances = find_anchors_across_time(vectors) #avg cos_dist over the years, per word (necessary for the diachronic anchor model)
    threshold, indices = find_nth_smallest(distances, pct_diachronic) #indices of diachronic anchors
    results, anchorIdxs = get_all_predictions(words, vectors, labels, indices, pct_anchors) #runs the three models
    write_results(results)
    

def read_data(data_folder='tsakalidis2019mining/'):
    '''
    Reads the data and returns a list of words, a list of labels and the word 
    embeddings per year.
    Note: In our experiments, we have used the labels released in our prior work.
    We have also selected to use strictly alphabetic words that appear in OED.
    To get the data, visit: https://zenodo.org/record/3383660#.Xp3Bz1MzafU
    '''
    #Read the interesection between TRI, SGNS and OED:
    intersected_file = 'interescted_vocab.txt'
    with open(intersected_file, 'r') as f:
        lines = f.readlines()
    f.close()
    intersected_vocabulary = [word[:-1] for word in lines if len(word)>=1]
    
    #First initialise words and labels
    print('Reading the words, the labels and the word vectors per year...')
    with open(data_folder+'word_labels.csv', 'r') as f:
        lines = f.readlines()
    f.close()
    
    words, labels, indices = [], [], []
    cnt = 0
    for line in lines[0:len(lines)-1]:
        fields = line[:len(line)-1].split(',')
        if fields[0] in intersected_vocabulary:
            words.append(fields[0])
            labels.append(fields[1])
            indices.append(cnt)
        cnt+=1
    fields = lines[len(lines)-1].split(',')
    if fields[0] in intersected_vocabulary:
        words.append(fields[0])
        labels.append(fields[1])
        indices.append(cnt)
        
    #Now initialise vectors (words are in the same order as in word_labels.csv)
    vectors = []
    for year in range(2000,2014):
        with open(data_folder+'vectors_'+str(year)+'.csv', 'r') as f:
            lines = f.readlines()
        f.close()
        
        wv = []
        cnt = 0
        for line in lines:
            if cnt in indices:
                fields = line.split(',')
                vec = np.array([np.float(fields[i]) for i in range(1, len(fields))])
                wv.append(vec)
            cnt+=1
        vectors.append(np.array(wv))
    vectors = np.array(vectors)
    
    return np.array(words), np.array(labels), np.array(vectors).transpose(1,0,2)

        
def find_anchors_across_time(vectors):
    '''
    Given the representations of words across time, it calculates the average
    displacement error per word, across time.
        - Input:   V-by-T-by-N vectors (V:vocab_size; T:num_timesteps; N:embed_dim)
        - Returns: V-dim vector (average displacement error of each across time)
    '''
    print('Calculating the alignment errors of the words through time...')
    errs = []
    for year in range(vectors.shape[1]-1):
        err = procrustes_find_distances(vectors[:,year,:], vectors[:,year+1,:])
        errs.append(err)
    return np.average(np.array(errs), axis=0)


def procrustes_find_distances(X, Y):
    '''
    (Based on Scipy)
    Performs Procrustes Alignment and calculates the per-word alignment error,
    measured by the cosine similarity.
        - Input:   two V-by-N matrices (V:vocab_size; N:embed_dim) to align
        - Returns: a single V-dim vector with the alignment errors per word
    '''
    muX, muY = np.average(X, axis=0), np.average(Y, axis=0)
    X0, Y0 = X - muX, Y - muY
    ssX, ssY = (X0**2.).sum(), (Y0**2.).sum()

    normX, normY = np.sqrt(ssX), np.sqrt(ssY)
    X0, Y0 = X0/normX, Y0/normY
    
    u, w, vt = np.linalg.svd(Y0.T.dot(X0).T)
    R = (u.dot(vt)).T
    s = w.sum()
    mtx = Y0.dot(R)*s

    errors = []
    for w in range(len(X0)):
        e = cos_dist(X0[w].reshape((1,-1)),mtx[w].reshape((1,-1)))[0, 0]
        errors.append(e)
    return np.array(errors)


def find_nth_smallest(dist, pctToKeep):
    '''
    Given a vector with distances and a % of words (of the vocab size) to keep
    as anchors, it returns the indices of the words with the lowest distances.
        - Input: 
            dist: V-dim vector of distances (V:vocab_size); 
            pctToKeep: percentage of words to keep as anchors (condition)
        - Returns: 
            threshold: the max distance score satisfying the condition; 
            indices: the indices of top-pctToKeep% words satisfying the condition. 
    '''
    numToKeep = int(pctToKeep*len(dist))
    threshold = np.partition(dist, numToKeep-1)[numToKeep-1]       
    indices = np.where(dist<=threshold)[0]
    return threshold, indices


def get_all_predictions(words, vectors, labels, static_indices, numToKeep):
    '''This is where all models run. Returns a dictionary with the results'''
    anchors, anchors_time, everything = dict(), dict(), dict()

    anchorIdxs = []
    print('year\tall\tanchor\tdiahcronic')  
    for year in range(vectors.shape[1]-1):
        X, Y = vectors[:,year+1,:], vectors[:,year,:]
        X, Y = vectors[:,year+1,:], vectors[:,0,:]
    
        b = procrustes_all_together(X, Y)
        c, anchorIdx = procrustes_anchor(X, Y, numToKeep)
        d = procrustes_diachronic(X[static_indices], Y[static_indices], X, Y)
        
        everything[2000+year+1] = np.array([words, labels, b])
        anchors[2000+year+1] = np.array([words, labels, c])
        anchors_time[2000+year+1] = np.array([words, labels, d])      
        anchorIdxs.append(anchorIdx)
    
        print(year, '\t', evaluate_murank(b, labels),'\t', evaluate_murank(c, labels), '\t', evaluate_murank(d, labels))
    results = dict()
    results['ANCHORS'] = anchors
    results['DIACHRONIC'] = anchors_time
    results['ALL'] = everything    
    return results, anchorIdxs


#procrustes aligning everything together (no anchors)
def procrustes_all_together(x_all, y_all):
    _, x_all, y_all, __, ___ = procrustes_align(x_all, y_all)
    scores = []
    for w in range(len(x_all)):
        scores.append(cos_dist(x_all[w].reshape((1,-1)),y_all[w].reshape((1,-1)))[0, 0])
    return np.array(scores)


#procrustes trained on anchors    
def procrustes_anchor(x_all, y_all, numToKeep):
    anchor_indices = procrustes_find_anchor(x_all, y_all, numToKeep)
    x_anchor = x_all[anchor_indices]
    y_anchor = y_all[anchor_indices]
    _, __, ___, transform, params = procrustes_align(x_anchor, y_anchor) #what we really need is {transform, params}
    #print(transform, params)
    x_test, y_test = procrustes_transform(x_all, y_all, transform, params)
    scores  = []
    for w in range(len(x_test)):
        scores.append(cos_dist(x_test[w].reshape((1,-1)),y_test[w].reshape((1,-1)))[0, 0])
    return np.array(scores), anchor_indices

    
#procrustes trained on diachronic anchors
def procrustes_diachronic(x_train, y_train, x_test, y_test):
    _, __, ___, transform, params = procrustes_align(x_train, y_train) #what we really need is {transform, params}
    x_test, y_test = procrustes_transform(x_test, y_test, transform, params)
    scores  = []
    for w in range(len(x_test)):
        scores.append(cos_dist(x_test[w].reshape((1,-1)),y_test[w].reshape((1,-1)))[0, 0])
    return np.array(scores)


def procrustes_align(X, Y):
    '''
    The core part of the alignment. Given two matrices of word representations,
    it aligns them and returns the appropriate matrices.
    '''
    muX, muY = np.average(X, axis=0), np.average(Y, axis=0)
    X0, Y0 = X - muX, Y - muY
    ssX, ssY = (X0**2.).sum(), (Y0**2.).sum()

    normX, normY = np.sqrt(ssX), np.sqrt(ssY)
    X0, Y0 = X0/normX, Y0/normY
    
    u, w, vt = np.linalg.svd(Y0.T.dot(X0).T)
    R = (u.dot(vt)).T
    s = w.sum()
    mtx = Y0.dot(R)*s
    
    err = np.sum(np.square(X0 - mtx))
    ops = {'rotation':R, 'scale':s}
    vals = {'meanY': muY, 'normY': normY, 'meanX': muX, 'normX': normX}
    return err, X0, mtx, ops, vals
 

def procrustes_find_anchor(X, Y, numToKeep):
    muX, muY = np.average(X, axis=0), np.average(Y, axis=0)
    X0, Y0 = X - muX, Y - muY
    ssX, ssY = (X0**2.).sum(), (Y0**2.).sum()
    normX, normY = np.sqrt(ssX), np.sqrt(ssY)
    X0, Y0 = X0/normX, Y0/normY
    
    u, w, vt = np.linalg.svd(Y0.T.dot(X0).T)
    R = (u.dot(vt)).T
    s = w.sum()
    mtx = Y0.dot(R)*s

    errors = []
    for w in range(len(X0)):
        e = cos_dist(X0[w].reshape((1,-1)),mtx[w].reshape((1,-1)))[0, 0]
        errors.append(e)
    errors = np.array(errors)
    threshold, indices = find_nth_smallest(errors, numToKeep)
    return indices
 
    
def procrustes_transform(x, y, t, params):
    z0 = (y - params['meanY'])/params['normY']
    x0 = (x - params['meanX'])/params['normX']
    mtx = t['scale']*z0.dot(t['rotation'])
    return x0, mtx


def write_results(results, outfolder='dynamic_results/'):
    for model in results.keys():
        model_data = results[model]
        for year in range(2001,2014):
            data = model_data[year]
            words, labels, scores = data[0, :], data[1, :], data[2, :]
            outfile = outfolder+model+'_'+str(year)+'.tsv'
            with open(outfile, 'w') as out:
                for i in range(len(words)):
                    out.write(str(words[i])+'\t'+str(labels[i])+'\t'+str(scores[i])+'\n')
            out.close()
            
            
def evaluate_murank(scores, labels):   
    z = list(zip(scores, labels))
    z.sort()
    z.reverse()
    scores, labels = zip(*z)
    scores, labels = np.array(scores), np.array(labels)
    changed_idx = np.where(labels=='change')[0]/len(scores)
    average_rank = 100*np.average(changed_idx)
    return np.round(average_rank,2)


def evaluate_recall(scores, labels):
    z = list(zip(scores, labels))
    z.sort()
    z.reverse()
    scores, labels = zip(*z)
    scores, labels = np.array(scores), np.array(labels)
    labels = labels[:int(0.1*len(scores))]
    rec = 100*len(np.where(labels=='change')[0])/65
    return np.round(rec,2)


def evaluate(model_name='ALL', metric='murank'):
    folder = 'dynamic_results/'
    muranks, recalls = [], []
    
    for year in range(2001, 2014):
        infile = folder+model_name+'_'+str(year)+'.tsv'
        scores, labels = [], []
        with open(infile, 'r') as f:
            lines = f.readlines()
        for line in lines:
            fields = line.split('\t')
            labels.append(fields[1])
            scores.append(np.float(fields[2]))
        recalls.append(evaluate_recall(scores, labels))
        muranks.append(evaluate_murank(scores, labels))       
    print(np.average(muranks), np.average(recalls))
    
    
def print_words(model_name):
    infile = 'dynamic_results/'+model_name+'_2013.tsv'
    with open(infile, 'r') as f:
        lines = f.readlines()
    words, labels, scores = [], [], []
    for line in lines:
        fields = line.split('\t')
        words.append(fields[0])
        labels.append(fields[1])
        scores.append(np.float(fields[2]))
    z = list(zip(scores, labels, words))
    z.sort()
    z.reverse()
    scores, labels, words = zip(*z)
    for i in range(len(scores)):
        if labels[i]=='change':
            print(words[i])