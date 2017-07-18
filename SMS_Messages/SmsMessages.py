#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:13:28 2017

@author: Anani Assoutovi
"""
# SMSSpamCollection
import pandas as pd, string, pprint
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
import visuals as vs
ds = pd.read_table('SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label','sms_messages'])
print('\n')
print(ds.head())
"""Our model would still be able to make predictions if 
    we left our labels as strings but we could have issues 
    later when calculating performance metrics, for example when 
    calculating our precision and recall scores. Hence, to avoid 
    unexpected 'gotchas' later, it is good practice to have our 
    categorical values be fed into our model as integers.
"""

"""
Convert the values in the 'label' colum to numerical values using 
map method as follows: {'ham':0, 'spam':1} This maps the 'ham' value 
to 0 and the 'spam' value to 1. Also, to get an idea of the size 
of the dataset we are dealing with, print out number of rows and 
columns using 'shape'.
"""
print('\n')
ds['label'] = ds.label.map({'ham':0, 'spam':1})
print(ds.shape)
print(ds.head())
#vs.survival_stats(ds, 'spam')
print('\n')

documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_document = []
for i in documents:
    lower_case_document.append(i.lower())
print(lower_case_document)

"""
Remove all punctuation from the strings in the document set. 
Save them into a list called 'sans_punctuation_documents'.
"""
print('\n')
sans_punctuation_document = []
for i in lower_case_document:
    sans_punctuation_document.append(i.translate(str.maketrans('','',string.punctuation)))
print(sans_punctuation_document)

"""
Tokenize the strings stored in 'sans_punctuation_documents' using the split() method. 
and store the final document set in a list called 'preprocessed_documents'.
"""
print('\n')
preprocessed_document = []
for i in sans_punctuation_document:
    preprocessed_document.append(i.split(' '))
print(preprocessed_document)


"""
Using the Counter() method and preprocessed_documents as the input, 
create a dictionary with the keys being each word in each document 
and the corresponding values being the frequncy of occurrence of that word. 
Save each Counter dictionary as an item in a list called 'frequency_list'.
"""
print('\n')
frequency_list = []
for i in preprocessed_document:
    frequency_count = Counter(i)
    frequency_list.append(frequency_count)
pprint.pprint(frequency_list)

"""
Import the sklearn.feature_extraction.text.CountVectorizer 
method and create an instance of it called 'count_vector'.
"""
print('\n')
count_vector = CountVectorizer()
#print(count_vector)

"""
Fit your document dataset to the CountVectorizer object you 
have created using fit(), and get the list of words which have 
been categorized as features using the get_feature_names() method.
"""
print('\n')
documents = ['Hello, how are you!',
                'Win money, win from home.',
                'Call me now.',
                'Hello, Call hello you tomorrow?']

count_vector.fit(documents)
count_vector.get_feature_names()
print(count_vector.get_feature_names())

"""
Create a matrix with the rows being each of the 4 documents, 
and the columns being each word. The corresponding (row, column) 
value is the frequency of occurrance of that word(in the column) 
in a particular document(in the row). You can do this using the 
transform() method and passing in the document data set as the argument. 
The transform() method returns a matrix of numpy integers, you can convert 
this to an array using toarray(). Call the array 'doc_array'
"""
print('\n')
doc_array = count_vector.transform(documents).toarray()
print(doc_array)

"""
Convert the array we obtained, loaded into 'doc_array', 
into a dataframe and set the column names to the word 
names(which you computed earlier using get_feature_names(). 
Call the dataframe 'frequency_matrix'.
"""
print('\n')
frequency_matrix = pd.DataFrame(doc_array,
                               columns = count_vector.get_feature_names())
print(frequency_matrix)

"""
Split the dataset into a training and testing set by using the 
train_test_split method in sklearn. Split the data using the following 
variables:
X_train is our training data for the 'sms_message' column.
y_train is our training data for the 'label' column
X_test is our testing data for the 'sms_message' column.
y_test is our testing data for the 'label' column Print out the 
number of rows we have in each our training and testing data.
"""

print('\n')
X_train, X_test, Y_train, Y_test = train_test_split(ds['sms_messages'],
                                                    ds['label'],
                                                    random_state=1)
print('Number of rows in total set: {}'.format(ds.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))
print('\n\n')

#Instantiate the CounterVectorizer method
count_Vector = CountVectorizer()

##Fit the training data and then return the matrix
training_data = count_Vector.fit_transform(X_train)

## Transform Testing data and then return the matrix. Note we are not fitting the testing data in our 
## CountVectorizer()
testing_data = count_Vector.transform(X_test)

print('Training Data: {}'.format(training_data))
print('\nTesting Data: {}'.format(testing_data))

print('\n')







