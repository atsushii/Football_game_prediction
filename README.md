# Spam-Classification

# Overview

This project is going to predict spam or ham from SMS text data.
As you know many company uses this ML technichs to avoid spam message such as gmail.

I will use several ML algorythms and Neural network to predict spam SMS also using NLP technich for text processing.

I use a SMS dataset from kaggle.

Dataset: https://www.kaggle.com/uciml/sms-spam-collection-dataset

# Text processing

Dataset is including text data, we need to normalize it also need a converting into numerical value.
This process I used a NLP technichs such as nltk and TF-IDF.


  **Convert to lowercase**
  The reason is if you don't convert to lowercase all of sentences, Tha model might treat a defferent word between capital
  letter and lowercase even if these words are same.

  e.g. 
  
  Hello and hello → hello and hello
  

  **Remove punctuation**
  it doesn't affect to predict spam SMS without punctuation. But some NLP tasks need to use a puncutuation such as machine 
  translation.
  I remove a punctuaiton but number of punctuation of each sentence is going to be feature.
  It might be affected to prediction.
  

  **Remove stop word**
  I remove a stop word but it depends on detaset. For instance your goal is sentiment analysis from text then if you remove   
  stop words from text,

  e.g.
  
  I was not happy. → [happy]

  The result might be positive but actuall result is negative because sentence is "I was not happy"
  So removing stop word is not always make a good result.
  In classification problem usually no need to leave stop words in text. Because able to understand general idea without 
  stopwords.
  

  **Stemming**
  Stemmig is a process that words are reduced to root by removing unnessesary charactores.
  Usualy suffix.

  e.g.

  playing → play

  plays → play
  
  Without steeming, model can not understund play and plays are same word, but after stemming each words will be root word.
  The model can recognize ezxh words are same word.
  

# Feature extraction

Featuer extraction is process of dimentional reduction by which an initial set of raw data is reduced to more manageble group for processing. Also transforming data such as image or text, into numerical features.
I use a TF-IDF to convert text to numerical features.

  **TF-IDF**
  
  ML is not working text data directry, we need to convert into vectors of number.
  
  
  
  
  

  






