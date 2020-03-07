# Spam-Classification

# Overview

This project is going to predict spam or ham from SMS text data.
As you know many company uses this ML techniques to avoid spam message such as Gmail.

I will use several ML algorithms and Neural network to predict spam SMS also using NLP technique for text processing.

I use an SMS dataset from kaggle.

Dataset: https://www.kaggle.com/uciml/sms-spam-collection-dataset

# Text processing

Dataset is including text data, we need to normalize it also need a converting into a numerical value.
In this process I used a NLP techniques such as nltk and TF-IDF.


  **Convert to lowercase**
  The reason is if you don't convert to lowercase all of sentences, Tha model might treat a different word between capital
  letter and lowercase even if these words are the same.

  e.g. 
  
  Hello and hello → hello and hello
  

  **Remove punctuation**
  
  
  it doesn't affect to predict spam SMS without punctuation. But some NLP tasks need to use punctuation such as machine 
  translation.
  I remove punctuation but a number of punctuation of each sentence is going to be feature.
  It might be affected by prediction.
  

  **Remove stop word**
  
  
  I remove a stop word but it depends on the dataset. For instance, your goal is sentiment analysis from text then if you 
  remove   
  stop words from text,

  e.g.
  
  I was not happy. → [happy]

  The result might be positive but actual result is negative because the sentence is "I was not happy"
  So removing stop words is not always make a good result.
  In classification problems usually no need to leave stop words in text. Because able to understand the general idea without 
  stopwords.
  

  **Stemming**
  
  
  Stemming is a process that words are reduced to root by removing unnecessary characters.
  Usually suffix.

  e.g.

  playing → play

  plays → play
  
  Without steeming, the model can not understand play and plays are the same word, but after stemming each word will be root 
  word.
  The model can recognize each word are the same word.
  

# Feature extraction

Feature extraction is the process of dimensional reduction by which an initial set of raw data is reduced to a more manageable group for processing. Also transforming data such as image or text, into numerical features.
I use a TF-IDF to convert text to numerical features.

  **TF-IDF**
  
  ML is not working text data directory, we need to convert into vectors of number.
  TF-IDF can calculate score the relative importance of words.
  
  Simply TF-IDF = TF * IDF.
  
  ![Untitled (18)](https://user-images.githubusercontent.com/25543738/75927724-c3cac900-5e21-11ea-8008-8b8a611aff58.png)

  
  I will explain what is TF and IDF.
    
   
   **Term Frequency (TF)**
   
   This is calculate how many times a word appears in a document by a total number of words in a document.
   Tf is often used in Text Mining.
   Every documents are different length, it means long sentence has term appear many times.
   We can normalize word frequency below formula.

   TF = (Number of times word t appears in a document / Total number of term in a document)


   ![Untitled (16)](https://user-images.githubusercontent.com/25543738/75927820-f70d5800-5e21-11ea-8004-d103269ad4b7.png)


   
   **Inverse Document Frequency (IDF)**
   
   IDF is a weight indicating how commonly a word is used. The more frequent word is lower score. In other words
   this word is not important in a document.
   IDF is often used to boost the score of words that is unique in a document. We can find more influential word in  
   document.

   IDF = log(Total number of documents / Number of documents containing the word t)


   ![Untitled (17)](https://user-images.githubusercontent.com/25543738/75927772-dd6c1080-5e21-11ea-82c0-81a5b631051c.png)
   
   
# Machine Learning Algorithme

I use different algorithms to train models.

# Logistic Regression
 
 Logistic Regression is used to solve classification problems.
 In this project, the target value is categorical.
 
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
  
  When we train the model, We need to maximize the probability by minimizing a loss function.
  Cost function will be a convex function of the parameter, Gradient descent will converge into global minimum.
 
 ![Untitled (21)](https://user-images.githubusercontent.com/25543738/75947154-453b4f00-5e54-11ea-9fa2-d22b18a571c7.png)
 
 
 # XGBoost
 
 XGBoost is a decision-tree-based ensemble algorithm, using a gradient boosting framework.
 It is focus on model performance and computational speed.
 XGBoost can be used to solve regression, classification and ranking problems.
 
 # Neural network
 
 Neural network has a multi-layer network of neurons.
 Usually having an input layer, hidden layer, and output layer.
 we can add n hidden layers into network.
 
 ![Untitled (23)](https://user-images.githubusercontent.com/25543738/75950624-918b8c80-5e5e-11ea-894d-32871a8f69c2.png)
 
 The input layer is connected with the hidden layer, each connection has numerical value as weight.
 Then input is multiplied to corresponding the weights. Then sum it into neuron in the hidden layer, each neuron has a 
 numerical value called bias. Then this value passing to the activation threshold function called activation function.
 I use sigmoid which is already explained.
 The result of the activation function determines if neurons will get activated or not.
 activated neurons are moving to the output layer.
 This method data is propagated through the network called forward propagation.
 
 Output value is basic probability, but if network has made the wrong prediction.
 During training, predict output compared to the actual value, then calculate an error(actual value - predict output)
 The magnitude of error indicates how wrong we are. Then give an indication of the direction and magnitude of change to reduce 
 the error. this information transferred backward through the network this is called back propagation.
 Based on this information weights are adjusted.
 forward propagation and back propagation iteratively performed during training.
 
 # Model Evaluation
 
 After train the model, We need to evaluate how the model works well.
 It is quite an important task in the real world.
 There are a few model evaluation techniques.
 
   **Confusion Matrix**
   
   It is a matrix representation of the results of any binary testing.
   
   
   Accuracy = (TP + TN) / (TP + TN + FP + FN)

   → all of the result, how much we predicted correctly, 
    
   Precision = TP / (TP + FP)
   
   → measure of the accuracy of your model, how many are actual positiv3
   
   Recall = TP / (TP + FN)
   
   → how much we predict correctly in the actual result it should be high.
   
   F-measure = 2 * Recall * Precision /  (Reacall + Precision)
   
   →  F-measure is used when the False Negatives and False Positives are crucial
   
   
   ![Untitled (24)](https://user-images.githubusercontent.com/25543738/76054183-afbbc000-5f24-11ea-9c88-74d6a6fd7151.png)
   
   In this project, I predict a spam SMS.
   Let's say 100 people receive an SMS. In actual 40 people got a spam SMS but you predict 15 people receive a spam SMS and 
   3 of them don't receive a spam SMS. it confusing, I will put the result into matrix.
   
   
   ![Untitled (25)](https://user-images.githubusercontent.com/25543738/76057402-4d67bd00-5f2e-11ea-9e9f-dd5d8c3e8e47.png)
   
   True Positive 12 (you have a positive case correct prediction)
   
   True Negative 57 (you have a negativr case correct prediction)
   
   False Positive 3 (you predicted 3 people recieve spam SMS but actual they don't)
   
   True negative 28 (you predicted 28 people don't recieve spam SMS but actual they recieve)
   
   **AUC-ROC Curve**
   
   This method is also useful for the evaluation of the model.
   
   ROC is ratio of True positive(TPR) and False positive(FRP)
   it shows how much model is capable of distinguishing between class.
   Higher AUS is a better model.
   
   ![Untitled (26)](https://user-images.githubusercontent.com/25543738/76061325-1eeedf80-5f38-11ea-83fa-b62a1d5dd93b.png)

   
   High-performance model has AUC near to the 1, A poor model has AUC near to the 0
   
   
   

   
   
   

   
   
   
   
   
   
 
 
 
 
 
 
 
 
 

 


 
 
 
 
 

 
 
 
 

 
 
 
 
    
    
    
  
  
  
  
  

  






