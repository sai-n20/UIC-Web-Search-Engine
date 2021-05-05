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