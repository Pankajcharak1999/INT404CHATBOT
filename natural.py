from newspaper import Article
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
import warnings

#ifnore any warning messages
warnings.filterwarnings('ignore')

#download the packages from nltk
nltk.download('punkt',quiet=True)
nltk.download('wordnet',quiet=True)

article = Article('https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521')
article.download()
article.parse()
article.nlp()
corpus = article.text

#print the corpus/text
# print(corpus)

#Tokenization
text = corpus
#convert the text into a list of sentence
sent_tokens = nltk.sent_tokenize(text)

#print the list of sentence
# print(sent_tokens)

# create a doictionary(key:value) pair to remove punctuations
remove_punct_dict = dict( (punct,None) for punct in string.punctuation)

#print the punctuations
# print(string.punctuation)

# print(remove_punct_dict)

#create a function to return a list of lemmatized lower case words after removing punctuations
def LemNormalize(text):
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))

#print the tokenizaation text
# print(LemNormalize(text))

#keyword matching
#Greetin inputs
greeting_inputs = ["hi","hello","greetings","wassup","hey"]
#greetind response back to the user
greeting_response = ["howdy","hi","hey","what's good","hello","hey there"]
#function to return a random greeting response to a user greeting
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_response)

#generate the response
def response(user_response):
    #The users response/query
    user_response = 'what is a chronic disease'
    #Make the response lower case
    user_response = user_response.lower()
    #print the user response
     # print(user_response)

    #set the chatbot response to an empty string
    robo_response = ""

    #append the user response to the sentense list
    sent_tokens.append(user_response)

     # print(sent_tokens)

    #create a TfidfVectorizer object
    TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words='english')

    #convert the text to a matrix of Tf-IDF features
    tfidf = TfidfVec.fit_transform(sent_tokens)

    #print the TFIDF features
    # print(tfidf)

    #get the measure of similarity
    vals = cosine_similarity(tfidf[-1], tfidf)

    #print the similarity scores
    # print(vals)

    #get the index of the most similar text/sentence to the users response
    idx = vals.argsort()[0][-2]

    #reduce the dimensionality off vlas
    flat = vals.flatten()

    #sort the list in ascending order
    flat.sort()

    #get the most similar score to the users response
    score = flat[-2]
    #print the similarity score
    # print(score)

    #if the variable score is 0 then their is no text similar to the user response
    if(score == 0):
        robo_response = robo_response+"I apologize, I don't understand."
    else:
        robo_response = robo_response+sent_tokens[idx]

    #print the chat bot response
    #print(robo_response)

    #remove the users reponse form the sentence token list
    sent_tokens.remove(user_response)


    return robo_response

flag = True
print("DOCBot: I am Doctor Bot or DOCBot for short. I will answer your queries about the chronic disease.If you need any help we here to response")
while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("DOCBot: You are welcome !")
        else:
            if(greeting(user_response)!=None):
                print("DOCBot: "+greeting(user_response))
            else:
                print("DOCBot: "+response(user_response))
    else:
        flag = False
        print("DOCBot: Chat with you later !")
