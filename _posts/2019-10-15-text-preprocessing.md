---
title: "Working with open-ended survey responses: text preprocessing"
date: 2019-09-06
header:
  image: /assets/images/post5/background.jpg
  teaser: /assets/images/post5/background.jpg
  caption: image from web
tags:
- survey
- text mining
- string
- wordcloud
- open-ended
- stemming
- lemmatization
- text preprocessing
categories:
- survey
- text
---

Following the [last post](https://rosalynnyang.github.io/survey/text/examine-text-open-end/), I will further discuss normal text preprocessing steps in Python.


```python
import numpy as np
import pandas as pd
```


```python
df = pd.read_parquet('../output/df')
```

In this survey dataset, there was a question that asked "Would you be willing to bring up a physical health issue with a potential employer in an interview?" and respondents were asked a follow up question explaining why they would or would not.


```python
# Question: Would you be willing to bring up a physical health issue with a potential employer in an interview?
df['physical_interview'].value_counts()
```




    Maybe    633
    No       441
    Yes      359
    Name: physical_interview, dtype: int64



It is normal to see missing data in open-ended questions since respondents tend to skip questions that require more energy (e.g. thinking and typing), but it would also depend on the type of the survey and how salient respondents are to the open-ended questions.


```python
len(df['physical_interview_y'].dropna().to_list())
```




    1095



After excluding missing data, we have 1095 text responses out of n=1433 respondents - not bad! Let's take a look at what some responses are and have a basic idea:


```python
df['physical_interview_y'].dropna()[0:15]
```




    1     It would depend on the health issue. If there ...
    2     They would provable need to know, to Judge if ...
    3     old back injury, doesn't cause me many issues ...
    4     Depending on the interview stage and whether I...
    5     If it would potentially affect my ability to d...
    6     I want to gauge their ability to support this ...
    7                               I feel it's irrelevant.
    8                 Makes me a less attractive candidate.
    9     Generally speaking, and this isn't always the ...
    10    Being honest upfront shows respect for the fut...
    11     It isn't relevant to my ability as a programmer.
    12                 Seems highly unlikely to be relevant
    13    I might have special needs that would be impos...
    14          Because of the potential for discrimination
    15    I don't think it's appropriate to discuss thos...
    Name: physical_interview_y, dtype: object



Alright. Most responses seem long, which makes sense since this is asking "why". Some mentioned it depends, some said it's irrelevant, and some others wanted to just be honest. <br>
The output display does not show the full sentence and it makes me wonder how long these responses usually are. We could quickly check number of words/length of the responses:


```python
text_df = df[df['physical_interview_y'].notnull()]
text_df = text_df[['ID', 'physical_interview', 'physical_interview_y']]
```


```python
# count number of words
text_list = text_df['physical_interview_y'].astype(str).to_list()
num_words = []
for text in text_list:
    num_words.append(len(text.split()))
```


```python
np.mean(num_words)
```




    16.122374429223743




```python
np.std(num_words)
```




    18.133084922078474



On average, each response is around 16 words but there is a high variance associated with it. You could imagine some people would just type up a few words while some others write up a personal story. <br>
Now, if my goal is to study why people would support or oppose this idea of bringing up one's physical conditions in job interviews, what should I do?

#### Text cleaning/preprocessing

The first step I take dealing with text is always preprocessing, no matter whether I am using string distance, topic models, or directly using a mature language model's word embeddings. The purpose of text preprocessing is to get the text ready for your analysis question. Therefore, preprocessing steps are usually analysis-specific. <br>
Since open-ended answers are quite verbal, I usually remove punctuations, numbers, common stop words, and then lowercase everything (though this could affect meanings of some words). <br>
There are a number of Python libraries that could help with text preprocessing, here I'll illustrate the basics with NLTK. 


```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
```


```python
# pull the list of stop words from NLTK corpus
stop_words = stopwords.words('english')
len(stop_words)
```




    179




```python
stop_words[:10]
```




    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]



There are 179 stop words in the NLTK stopword list with 10 examples displayed above. These are common English words that probably won't add much meaning to the responses we are looking into. We could also add more words to the list (that we want to ignore) using `.extend` method.


```python
customStopWords = ["couldn't","could", "wouldn't", "would", "can", "can't", "do", "don't", 
                   "wouldnt", "couldnt", "cant", "dont", 'im', 'none']
stop_words.extend(customStopWords)
```

As mentioned above, for these open-ended responses, I'd like to perform a series of preprocessing steps including removing punctuations, numbers, stop words, and tokenization. I created a simple function to realize this pipeline for each record of response: 


```python
def text_cleaner(text):
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    text = text.translate(str.maketrans('', '', string.digits)) # remove numbers
    tokens = word_tokenize(text) # tokenization
    tokens = [word.lower() for word in tokens] # normalise to lower case
    tokens = [word for word in tokens if not word in stop_words] # remove common stop words
    return tokens
```

Then we could apply this function to the all the responses in ths list:


```python
cleaned_text = []
for text in text_list:
    cleaned_text.append(text_cleaner(text))
```

We can pick a response and examine what it looks like before and after pre-processing:


```python
# before pre-processing
text_list[3]
```




    'Depending on the interview stage and whether I required an accommodation, I would'




```python
# after pre-processing
cleaned_text[3]
```




    ['depending', 'interview', 'stage', 'whether', 'required', 'accommodation']



#### Wordcloud

Let's use the good old wordcloud to visualize what's mentioned in these responses:


```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
```


```python
alltext_list = sum(cleaned_text,[])
alltext = " ".join(str(x) for x in alltext_list)
```


```python
wordcloud = WordCloud(background_color='white', collocations=False, width=800, height=400, prefer_horizontal=1).generate(alltext)
plt.figure(figsize=(12,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```


![png](/assets/images/post5/output_34_0.png)


This wordcloud probably doesn't reveal much since a lot of the words tend to mean similar things. However, we could notice other things from this visualiztion such as some more stop words to include (e.g. "may", "something") and consider stemming and/or lemmatization (e.g. hiring vs. hired). 

#### Stemming and Lemmatization

Stemming and lemmatization sometimes are mentioned as last steps in a text preprocessing pipeline. In short, stemming helps transform a word to its root form and lemmatization helps reduce the words to a word existing in the language. In other words, the result of stemming doesn't neccessarily is a word we recognize. Lemmatization is usually preferred as it results in real words, however, it requires linguistic analysis/more computing power and part-of-speech (pos) tagging.

I'll cover an example: 


```python
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
```


```python
ps = PorterStemmer()
wnl = WordNetLemmatizer()
```

With stemming, we could see that the word "depending" has been truncated as "depend", the word "required" back to "requir" (not a word anymore).


```python
[ps.stem(word) for word in cleaned_text[3]]
```




    ['depend', 'interview', 'stage', 'whether', 'requir', 'accommod']



With lemmatization, there doesn't seem to be any change without specifying the part-of-speech tagging.


```python
[wnl.lemmatize(word) for word in cleaned_text[3]]
```




    ['depending', 'interview', 'stage', 'whether', 'required', 'accommodation']



For example, the word "depending" could be lemmatized into "depending" when it's specificed as a noun:


```python
print(wnl.lemmatize("depending", pos='n'))
```

    depending
    

When it's a verb, it would be lemmatized into "depend":


```python
print(wnl.lemmatize("depending", pos='v'))
```

    depend
    

Stemming and lemmatization are techniques that are even more specific to the analysis question. I tend to not use them in my preprocessing steps unless needed to. 

In the next post, I will discuss how to transform text into numbers and run basic topic models to understand what's mentioned in the responses.
