#!/usr/bin/env python
# coding: utf-8

import pickle
import os
import nltk
import re
import numpy as np
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_list = stopwords.words('english')
ps = PorterStemmer()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


crawledList = []
pageTitles = []
webData = []

with open('pageList.txt', "rb") as out:
    crawledList = pickle.load(out)
out.close()

with open('pageTitles.txt', "rb") as out:
    pageTitles = pickle.load(out)
out.close()


with open('webData.txt', "rb") as out:
    webData = pickle.load(out)
out.close()

with open('page_rank.pkl', 'rb') as out:
    pageRank = pickle.load(out)
out.close()


def cleaner(data):
    cleanData = []
    for file in data:
        temp = file.split()
        temp = [ps.stem(word) for word in temp]
        temp = [item for item in temp if item not in stop_list]
        temp = [item for item in temp if len(item) > 2]
        temp = (" ").join(temp)
        for i in punctuation:
            temp = temp.replace(i, "")
        cleanData.append(temp)
    return cleanData


print("Cleaning and pre-processing webpage data, this may take 20-30 seconds")
cleanWebData = cleaner(webData)
print("Webpage data pre processing complete.")


vectorizer = TfidfVectorizer()
tfidfs = vectorizer.fit_transform(cleanWebData)


def queryTokenize(query):
    temp = query.lower()
    temp = temp.split()
    temp = [ps.stem(word) for word in temp]
    temp = [item for item in temp if item not in stop_list]
    temp = [item for item in temp if len(item) > 2]
    temp = (" ").join(temp)
    for i in punctuation:
        temp = temp.replace(i, "")
    return temp.split()


def correlateTitleURL(length):
    pageNums = []
    for i in range(length):
        pageNums.append([int(bestFirstList[i][1]), 0.4 * bestFirstList[i][0] + 0.6 * pageRank.get(crawledList[int(bestFirstList[i][1])])])
    
    pageNums = sorted(pageNums, key= lambda x: x[1], reverse=True)
    
    for item in range(length):
        print(item + 1, "-" + pageTitles[pageNums[item][0]] + "\nhttps://" + crawledList[pageNums[item][0]] + "\n")


query = str(input("Enter a search query: "))
print('\n')
queryTokens = queryTokenize(query)
queryVector = vectorizer.transform([' '.join(queryTokens)])
simScores = cosine_similarity(tfidfs, queryVector)

bestFirstList = []
for i in range(len(cleanWebData)):
    bestFirstList.append((simScores[i][0], i))

bestFirstList = sorted(bestFirstList, key= lambda x: x[0], reverse=True)

searcher = True
toFetch = 10

while searcher or morePages.lower() in {'y', 'yes'}:
    searcher = False
    correlateTitleURL(toFetch)
    morePages = str(input("\nFetch more pages?"))
    toFetch += 10



#Code for manually creating TF-IDF dict and vectorizing corpus+query. 
#Decided to use sklearn implementation for conciseness and speed upon query by user.

# #Word freq dict and subsequent list
# tokenFreq = {}
# for page in cleanWebData:
#     for word in page.split():
#         if(word in tokenFreq):
#             tokenFreq[word] += 1
#         else:
#             tokenFreq[word] = 1
# vocab = [term for term in tokenFreq]

# #Create tfIDF dict of tuples
# tfIDF = {}
# for number, doc in enumerate(cleanWebData):
#     docTokens = doc.split()
#     wordCounts = Counter(docTokens)
#     totalWords = len(docTokens)
#     for word in np.unique(docTokens):
#         tf = wordCounts[word]/totalWords
#         df = tokenFreq[word]
#         idf = np.log((len(docCorpus))/df)

#         tfIDF[number, word] = tf*idf

# #Document vector from tfIDF
# docVector = np.zeros((len(cleanWebData), len(vocab)))
# for item in tfIDF:
#     index = vocab.index(item[1])
#     docVector[item[0]][index] = tfIDF[item]

# #Vectorize query in the same shape as document vector
# def vectorizeQuery(data):
#     vecQ = np.zeros((len(vocab)))
#     wordCounts = Counter(data)
#     totalWords = len(data)
    
#     for token in np.unique(data):
#         tf = wordCounts[token]/totalWords
#         df = tokenFreq[token] if token in vocab else 0
#         idf = np.log((len(cleanWebData)+1)/(df+1))
#         try:
#             ind = vocab.index(token)
#             vecQ[ind] = tf*idf
#         except:
#             pass
#     return vecQ

# #Calculate cosine similarity and arrange docs in descending order
# def cosineSimilarity(query, k):
#     tokens = query.split()
#     cosines = []
#     queryVector = vectorizeQuery(tokens)
#     for item in docVector:
#         cosines.append(np.dot(queryVector, item)/(np.linalg.norm(queryVector)*np.linalg.norm(item)))
#     if k > 0:
#         # top k docs in descending order    
#         return np.array(cosines).argsort()[-k:][::-1]
#     else:
#         # consider all docs
#         return np.array(cosines).argsort()[::-1]