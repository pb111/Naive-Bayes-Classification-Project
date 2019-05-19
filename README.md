# Naïve Bayes Classifier with Python and Scikit-Learn

In this project, I build a Naïve Bayes Classifier to predict whether a person makes over 50K in a year. I implement Naive Bayes Classification with Python and Scikit-Learn. The problem is to predict whether a person makes over 50K in a year. To answer the question, I build a Naive Bayes classifier to predict the income of the person. I have used the **census income** data set for this project, downloaded from the UCI Machine Learning Repository website.


===============================================================================


## Table of Contents

I have categorized this project into various sections which are listed below:-

1.	Introduction to Naïve Bayes algorithm
2.	Bayes theorem
3.	Class and conditional probabilities
4.	Naïve Bayes algorithm intuition
5.	Types of Naïve Bayes algorithm
       -	Gaussian Naïve Bayes algorithm
       -	Multinomial Naïve Bayes algorithm
       -	Bernoulli Naïve Bayes algorithm
6.	Advantages and disadvantages of Naïve Bayes algorithm
7.	The problem statement
8.	Results and conclusion
9.	Applications of Naïve Bayes algorithm
10.	References




## 1. Introduction to Naïve Bayes Classification algorithm

In machine learning, Naïve Bayes classification is a straightforward and powerful algorithm for the classification task. Naïve Bayes classification is based on applying Bayes’ theorem with strong independence assumption between the features.  Naïve Bayes classification produces good results when we use it for textual data analysis such as Natural Language Processing.
Naïve Bayes models are also known as `simple Bayes` or `independent Bayes`. All these names refer to the application of Bayes’ theorem in the classifier’s decision rule.  
Naïve Bayes classifier applies the Bayes’ theorem in practice. This classifier brings the power of Bayes’ theorem to machine learning. So, to understand the Naïve Bayes classifier we need to understand the Bayes’ theorem. So, I will discuss the Bayes’ theorem in next section.

## 2. Bayes theorem

Bayes’ theorem is a very important theorem in the field of probability and statistics.  The Bayes’ theorem describes the probability of an event based on prior knowledge of conditions that might be related to the event.
Bayes’ theorem is stated mathematically as the following equation–

# D – Bayes’ theorem

where A and B are events and P (B) ≠ 0.

**Terminology regarding Bayes’ theorem**

-	P (A | B) is a conditional probability – the likelihood of event A occurring given that B is true.

-	P (A | B) is also known as Posterior probability of A given that event B has already occurred.

-	P (B | A) is also a conditional probability – the likelihood of event B occurring given that A is true.

-	P (A) and P (B) are the probabilities of observing A and B independently of each other. This is also known as the marginal or prior probability.




## 3. Class and Conditional Probabilities

In the Bayes’ theorem, P (A) represents the probabilities of each event. In the Naïve Bayes Classifier, we can interpret these as `Class Probabilities`. It is simply the frequency of each instance of the event divided by the total number of instances.
In the theorem, P (A | B) represents the conditional probabilities of an event A given another event B has already occurred. In the Naïve Bayes Classifier, it means the posterior probability of A occurring when B is true.



## 4. Naïve Bayes Classifier intuition

Naïve Bayes Classifier uses the Bayes’ theorem to predict membership probabilities for each class such as the probability that given record or data point belongs to a particular class. The class with the highest probability is considered as the most likely class. This is also known as the **Maximum A Posteriori (MAP) **. 

The **MAP for a hypothesis with 2 events A and B is**

**MAP (A)**
= max (P (A | B))
= max (P (B | A) * P (A))/P (B)
= max (P (B | A) * P (A))

Here, P (B) is evidence probability. It is used to normalize the result. It remains the same, So, removing it would not affect the result.
Naïve Bayes Classifier assumes that all the features are unrelated to each other. Presence or absence of a feature does not influence the presence or absence of any other feature. 
In real world datasets, we test a hypothesis given multiple evidence on features. So, the calculations become quite complicated. To simplify the work, the feature independence approach is used to uncouple multiple evidence and treat each as an independent one.





## 5. Types of Naïve Bayes algorithm
There are 3 types of Naïve Bayes algorithm. The 3 types are listed below:-
1.	Gaussian Naïve Bayes
2.	Multinomial Naïve Bayes
3.	Bernoulli Naïve Bayes
These 3 types of algorithm are explained below.
### Gaussian Naïve Bayes algorithm
When we have continuous attribute values, we made an assumption that the values associated with each class are distributed according to Gaussian or Normal distribution. For example, suppose the training data contains a continuous attribute x. We first segment the data by the class, and then compute the mean and variance of x in each class. Let µi be the mean of the values and let σi be the variance of the values associated with the ith class. Suppose we have some observation value xi . Then, the probability distribution of xi given a class can be computed by the following equation –
# D – Gaussian Naïve Bayes


### Multinomial Naïve Bayes algorithm
With a Multinomial Naïve Bayes model, samples (feature vectors) represent the frequencies with which certain events have been generated by a multinomial (p1, . . . ,pn) where pi is the probability that event i occurs. Multinomial Naïve Bayes algorithm is preferred to use on data that is multinomially distributed. It is one of the standard algorithms which is used in text categorization classification.

### Bernoulli Naïve Bayes algorithm
In the multivariate Bernoulli event model, features are independent boolean variables (binary variables) describing inputs. Just like the multinomial model, this model is also popular for document classification tasks where binary term occurrence features are used rather than term frequencies.



## 6. Advantages and disadvantages of Naïve Bayes algorithm
The advantages of Naïve Bayes algorithm are as follows:-
1.	Naïve Bayes is a fast, simple and easy to implement algorithm. So, it may outperform more complex models when we have limited amount of data.
2.	Naïve Bayes can be used for binary and multiclass classification.
3.	It is a great choice for text classification problems. It is a popular choice for spam email classification.
4.	It can be easily trained on small dataset.
5.	Naïve Bayes works well with numerical and categorical data. It can also be used to perform regression by using Naïve Bayes.
The disadvantages of Naïve Bayes algorithm are as follows:-
1.	Naïve Bayes algorithm involves the use of the Bayes theorem. So, it does not work well when we have particular missing values or missing combination of values.
2.	Naïve Bayes algorithm works well when we have simple categories. But, it does not work well when the relationship between words is important.



## 7. The problem statement
In this project, I try to make predictions where the prediction task is to determine whether a person makes over 50K a year. I implement Naive Bayes Classification with Python and Scikit-Learn. 
To answer the question, I build a Naive Bayes classifier to predict whether a person makes over 50K a year.
I have used the **Census income** data set for this project. I have downloaded this dataset from the UCI Machine Learning Repository website. The data set can be found at the following url:-
https://archive.ics.uci.edu/ml/datasets/Adult

## 8. Results and conclusion






## 9. Applications of Naïve Bayes classification
Naïve Bayes is one of the most straightforward and fast classification algorithm. It is very well suited for large volume of data. It is successfully used in various applications such as 

1.	Spam filtering
2.	Text classification
3.	Sentiment analysis
4.	Recommender systems
It uses Bayes theorem of probability for prediction of unknown class.

## 10. References

The work done in this project is inspired from following books and websites:-

1.	Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron
2.	Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido
3.	Udemy course – Machine Learning – A Z by Kirill Eremenko and Hadelin de Ponteves
4.	https://en.wikipedia.org/wiki/Naive_Bayes_classifier

5.	http://dataaspirant.com/2017/02/06/naive-bayes-classifier-machine-learning/
6.	https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn

7.	https://stackabuse.com/the-naive-bayes-algorithm-in-python-with-scikit-learn/

8.	https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html










