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
  TF-IDF can calculate score the reletive importance of words.
  
  Simply TF-IDF = TF * IDF.
  
  ![Untitled (18)](https://user-images.githubusercontent.com/25543738/75927724-c3cac900-5e21-11ea-8008-8b8a611aff58.png)

  
  I will explane what is TF and IDF.
    

   **Term Frequency (TF)**
   
   This is calculate how many times a word appeares in a document by total number of words in a document.
   Tf is offen used in Text Mining.
   Every documents are defferent length, it means long sentence has term appear many times.
   We can normalize word frequence below formula.

   TF = (Number of times word t appears in a document / Total number of term in a document)

   ![Untitled (16)](https://user-images.githubusercontent.com/25543738/75927820-f70d5800-5e21-11ea-8004-d103269ad4b7.png)


   
   **Inverse Document Frequency (IDF)**
   
   IDF is a weight indicating how commonly a word is used. The more frequence word is lower score. In other words
   this word is not important in a document.
   IDF is offen used to boost the score of words that is unique in a document. We can find more influential word in  
   document.

   IDF = log(Total number of documents / Number of documents containing the word t)

   ![Untitled (17)](https://user-images.githubusercontent.com/25543738/75927772-dd6c1080-5e21-11ea-82c0-81a5b631051c.png)
   
   
# Machine Learning Algorithme

I use different algorithmesn to train model.

# Logistic Regression
 
 Logistic Regression is used to solve classification problem.
 In this project, target value is categorical.
 
 **Sigmoid Function**
 
 I convert categorical value into numerical
 y∈{0,1} 0 is negative class, 1 is positive class. 
 
 ![Untitled (19)](https://user-images.githubusercontent.com/25543738/75945598-9268f200-5e4f-11ea-9ea5-ab89f226a414.png)
 
 Some people already notice that logistic regression uses sigmoid function.
 
 ![Untitled (20)](https://user-images.githubusercontent.com/25543738/75945654-b298b100-5e4f-11ea-8e60-87199c39b0be.png)
  
  Look at above, g(z) is mapped any real number to (0, 1) making it easy to transform arbitrary-valued function into a 
  function 
  better suited for classification.
  
  **Decision Boundary**
  
  To get output as 0 or 1, we need to set a threshold.
  
  e.g.
  predict value >= 0.5, then SMS is spam otherwise non spam.
  
  **Cost Function**
  
  When we train the model, We need maximize the probability by minimizing loss function.
  Cost function will be convex function of parameter, Gradient descent will converge into global minimum.
 
 ![Untitled (21)](https://user-images.githubusercontent.com/25543738/75947154-453b4f00-5e54-11ea-9fa2-d22b18a571c7.png)
 
 
 # XGBoost
 
 XGBoost is decision-tree-based ensemble algorithm, using gradient boosting framework.
 It is focus on model performance and computional speed.
 XGBoost can be used to solve regression, classification and ranking problem.
 
 # Neural network
 
 Neural network has multi layer network of neurons.
 Usually having input layer, hidden layer and output layer.
 we can add n hidden layers into network.
 
 I use a simple neural network like below.
 


 
 
 
 
 

 
 
 
 

 
 
 
 
    
    
    
  
  
  
  
  

  






