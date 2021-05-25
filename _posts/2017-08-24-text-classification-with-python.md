---
layout: post
title: Text Classification with Python
categories: [nlp]
comments: true
toc: false
---

> If you are already familiar with what text classification is, you might want to jump to [this part](#testing-the-algorithms), or get the code [here](https://github.com/joaorafaelm/text-classification-python).

## What is Text Classification?
Document or text classification is used to classify information, that is, assign a category to a text; it can be a document, a tweet, a simple message, an email, and so on.
In this article, I will show how you can classify retail products into categories. Although in this example the categories are structured in a hierarchy, to keep it simple I will consider all subcategories as top-level.

*If you are looking for complex implementations of large scale hierarchical text classification, I will leave links to some really good papers and projects at the [end](#conclusion) of this post.*

## Getting started
Now, before you go any further, make sure you have installed [Python3+](https://www.python.org/downloads/) and [virtualenv](https://virtualenv.pypa.io/en/stable/) *(optional, but I highly recommend you to use it)*.

Let's break down the problem into steps:
- [Setting up the environment](#setting-up-the-environment)
- [Gathering the data](#gathering-the-data)
- [Extracting features from the dataset](#extracting-features-from-the-dataset)
- [Testing the algorithms](#testing-the-algorithms)

## Setting up the environment
The main packages used in this projects are: [sklearn](http://scikit-learn.org), [nltk](http://www.nltk.org) and [dataset](https://dataset.readthedocs.io/en/latest/).
Due to the size of the data-set, it might take some time to clone/download the repository; NLTK data is also considerably big.
Run the following commands to setup the project structure and download the required packages:
```bash
# Clone the repo
git clone https://github.com/joaorafaelm/text-classification-python;
cd text-classification-python;

# Create virtualenv; skip this one if you dont have virtualenv.
virtualenv venv && source venv/bin/activate;

# Install all requirements
pip install -r requirements.txt;

# Download all data that NLTK uses
python -m nltk.downloader all;
```

## Gathering the data
The dataset that will be used was created by [scraping](https://en.wikipedia.org/wiki/Web_scraping) some products from Amazon. Scraping might be fine for projects where only a small amount of data is required, but it can be a really slow process since it is very simple for a server to detect a robot, unless you are rotating over a list of proxies, which can slow the process even more.

Using [this script](https://github.com/joaorafaelm/text-classification-python/blob/master/amazon_scrape.py), I downloaded information of over 22,000 products, organized into 42 top-level categories, and a total of 6233 subcategories. See the whole category tree structure [here](https://github.com/joaorafaelm/text-classification-python/blob/master/category_tree.txt).

Again, to keep it simple I will be using only 3 top-level categories: Automotive, Home & Kitchen and Industrial & Scientific. Including the subcategories, there are 36 categories in total.

To extract the data from database, run the command:

```bash
# dump from db to dumps/all_products.json
datafreeze .datafreeze.yaml;
```

Inside the project you will also find a file called [data_prep.py](https://github.com/joaorafaelm/text-classification-python/blob/master/data_prep.py), in this file you can set the categories you want to use, the minimum amount of samples per category and the depth of a category. As I said before, only 3 categories are going to be used: *Home & Kitchen, Industrial & Scientific and Automotive*. I did not specify the depth of the subcategories, but I did specify 50 as the minimum amount of samples (is this case, products) per category.
To transform the data dumped from the database into this "filtered" data, just execute the file:
```bash
python data_prep.py
```
The script will create a new file called **products.json** at the root of the project, and print out the category tree structure. Change the value of the variables `default_depth`, `min_samples` and `domain` if you need more data.

## Extracting features from the dataset
In order to run machine learning algorithms, we need to transform the text into numerical vectors. [Bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model) is one of the most used models, it assigns a numerical value to a word, creating a list of numbers. It can also assign a value to a set of words, known as [N-gram](https://en.wikipedia.org/wiki/N-gram).

Scikit provides a vectorizer called [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) which transforms the text based on the bag-of-words/n-gram model, additionally, it computes term frequencies and evaluate each word using the [tf-idf](https://en.wikipedia.org/wiki/Tf–idf) weighting scheme.

Counting terms frequencies might not be enough sometimes. Take the words 'cars' and 'car' for example, by only using *tf-idf*, they are considered different words. This problem can be solved using [Stemming](https://en.wikipedia.org/wiki/Stemming) and/or [Lemmatisation](https://en.wikipedia.org/wiki/Lemmatisation). And there is where [NLTK](http://www.nltk.org) comes into play.

NLTK offers some pretty useful tools for NLP. For this project I used it to perform *Lemmatisation* and [Part-of-speech tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging).

With *Lemmatisation* we can group together the inflected forms of a word. For example, the words 'walked', 'walks' and 'walking', can be grouped into their base form, the verb 'walk'. That is why we need to *POS tag* each word as a noun, verb, adverb, and so on.

It is also worth noting that some words despite the fact that they appear frequently, they do not really make any difference for classification, in fact they could even help misclassify a text. Words like 'a', 'an', 'the', 'to', 'or' etc, are known as [stop-words](https://en.wikipedia.org/wiki/Stop_words). These words can be ignored during the [tokenization](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html) process.

## Testing the algorithms
Now that we have all the features and labels, it is time to train the classifiers. There are a number of algorithms you can use for this type of problem, for example: Multinomial Naive Bayes, Linear SVC, SGD Classifier, K-Neighbors Classifier, Random Forest Classifier.
Inside the file [classify.py](https://github.com/joaorafaelm/text-classification-python/blob/master/classify.py) you can find an example using the SGDClassifier.
Run it yourself using the command:
```bash
python classify.py
```
It will print out the accuracy of each category, along with the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).

Here is how it is implemented:
load the dataset, initiate WordNetLemmatizer and PerceptronTagger from NLTK.
As I was only interested in nouns, verbs, adverbs and adjectives, I created a lookup dict to quicken up the process. Although NLTK is great, its aim is not performance, so I also implemented python's [LRU Cache](https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_Recently_Used_.28LRU.29) for both lemmatize and tagger functions.

```python
# Load data
dataset = json.load(open('products.json', encoding='utf-8'))

# Initiate lemmatizer
wnl = WordNetLemmatizer()

# Load tagger pickle
tagger = PerceptronTagger()

# Lookup if tag is noun, verb, adverb or an adjective
tags = {'N': wn.NOUN, 'V': wn.VERB, 'R': wn.ADV, 'J': wn.ADJ}

# Memoization of POS tagging and Lemmatizer
lemmatize_mem = lru_cache(maxsize=10000)(wnl.lemmatize)
tagger_mem = lru_cache(maxsize=10000)(tagger.tag)
```

Next, the tokenizer function was created. It breaks the text into words and iterate over them, ignoring the stop-words and POS-tagging/Lemmatising the rest. This function will receive all documents from the dataset.

```python
# POS tag sentences and lemmatize each word
def tokenizer(text):
    for token in wordpunct_tokenize(text):
        if token not in ENGLISH_STOP_WORDS:
            tag = tagger_mem(frozenset({token}))
            yield lemmatize_mem(token, tags.get(tag[0][1],  wn.NOUN))

```

At last the pipeline is defined; the first step is to call TfidfVectorizer, with the tokenizer function preprocessing each document, and then pass through the SGDClassifier.
The classifier is trained and tested using [10-fold Cross-Validation](http://statweb.stanford.edu/~tibs/sta306bfiles/cvwrong.pdf) provided by the [cross_val_predict](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) method from scikit-learn.

```python
# Pipeline definition
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        tokenizer=tokenizer,
        ngram_range=(1, 2),
        stop_words=ENGLISH_STOP_WORDS,
        sublinear_tf=True,
        min_df=0.00009
    )),
    ('classifier', SGDClassifier(
        alpha=1e-4, n_jobs=-1
    )),
])

# Cross validate using k-fold
y_pred = cross_val_predict(
    pipeline, dataset.get('data'),
    y=dataset.get('target'),
    cv=10, n_jobs=-1, verbose=20
)

# Print out precison, recall and f1 scode.
print(classification_report(
    dataset.get('target'), y_pred,
    target_names=dataset.get('target_names'),
    digits=3
))
```
And here are the accuracy results for each algorithm I tested (all algorithms were tested with their default parameters):

| Algorithms        | Precision  | Recall  |
| :---------------: | ---------: | ------: |
| SGDClassifier     |      0.975 |   0.975 |
| LinearSVC         |      0.972 |   0.971 |
| RandomForest      |      0.938 |   0.936 |
| MultinomialNB     |      0.882 |   0.851 |

The *precision* is the percentage of the test samples that were classified to the category and actually belonged to the category.

The *recall* is the percentage of all the test samples that originally belonged to the category and in the evaluation process were correctly classified to the category.

## Conclusion
As the category tree gets bigger, and you have more and more data to classify, you cannot use a model as simple as the one above (well, you can but its precision will be very low, not to mention the computational cost). Another important thing to notice, is how you structure the categories, in amazon category structure, a lot of subcategories are so confused that I doubt even humans could correctly classify products to them.
The full code of this post can be found [here](https://github.com/joaorafaelm/text-classification-python).

If you noticed something wrong, or you know something that can make the algorithms better, please do comment bellow. Thanks for reading!

## Further reading
- [Classifier Statistics](https://monkeylearn.com/docs/article/classifier-statistics/)

- [A Meta-Top-Down Method for Large-Scale Hierarchical Classification](http://ieeexplore.ieee.org/document/6522404/?reload=true)

- [A survey of hierarchical classification across different application domains](https://link.springer.com/article/10.1007/s10618-010-0175-9)

- [Hierarchical Text Categorization and Its Application to Bioinformatics](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.130.4074&rep=rep1&type=pdf)

- [Comparing Several Approaches for Hierarchical Classification of Proteins with Decision Trees](https://link.springer.com/chapter/10.1007/978-3-540-73731-5_12)

- [Tokenizing Words and Sentences with NLTK](https://pythonprogramming.net/tokenizing-words-sentences-nltk-tutorial/)

- [Natural Language Processing with Deep Learning](https://www.youtube.com/watch?v=OQQ-W_63UgQ)

- [Document Classification using Multinomial Naive Bayes Classifier](https://www.3pillarglobal.com/insights/document-classification-using-multinomial-naive-bayes-classifier)
