---
title: "Working with survey data: open-ended text answers"
date: 2019-09-06
header:
  image: /assets/images/post4/background.jpg
  teaser: /assets/images/post4/background.jpg
  caption: image from web
tags:
- survey
- text
- string
- wordcloud
- open-ended
categories:
- survey
- text
---

Researchers have mixed feelings towards the use of open-ended questions and their dislikes partially come from the difficulty of post-processing text data that they collected from the questions. In this post and the next, I'll illustrate how to examine and deal with short text answers in Python. <br>
If you think about what open-ended questions usually ask, it could be unaided recalls (e.g. what's in you mind when you think of XX), asking the whys (e.g. you mentioned you XX, can you tell me why?), or short answers responding to other-please specify. There are various types of open-ended questions, and it's better to choose the right tool to deal with them for efficiency. <br>
I will keep using the same survey dataset that I have been using for my previous posts in this series, feel free to check them out here: [post1](https://rosalynnyang.github.io/survey/checking-survey-data/), [post2](https://rosalynnyang.github.io/survey/variable-type-missing/), [post3](https://rosalynnyang.github.io/survey/survey-data-cleaning/).


```python
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
```


```python
df = pd.read_parquet('../output/df')
```

In this dataset, it's interesting to see that `gender` is not collected by a multiple question that we are familiar with. Instead of asking respondents to choose from a list of standard options such as `Male`, `Female`, this study used an open-ended text field that collected respondents' text descriptions of their gender. <br>

#### Examine short text answers


```python
len(df['gender'].unique())
```




    71




```python
df['gender'].value_counts().head(5) # display the top 5
```




    Male      610
    male      249
    Female    153
    female     95
    M          86
    Name: gender, dtype: int64




```python
df['gender'].value_counts().tail(5) # display the bottom 5
```




    Genderfluid (born female)       1
    Male (trans, FtM)               1
    Other/Transfeminine             1
    Bigender                        1
    Female or Multi-Gender Femme    1
    Name: gender, dtype: int64



So looks like there were 71 kinds of responses, including variations of gender descriptions and spelling/lower/upper casing kind of variations. I'll first transform all text into lowercase and create a wordcloud to quickly visualize it. <br>
Note it's worth considering whether making all texts into lowercase/uppercase is a good choice for your text since some nuances could be captured by the capitalization of letters. In my case here, I wouldn't mind changing all texts to lower case.


```python
df['gender'] = df['gender'].str.lower()
```


```python
gender_list = df['gender'].to_list()
text = " ".join(str(x) for x in gender_list)
```


```python
wordcloud = WordCloud(background_color='white').generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```


![png](/assets/images/post3/output_13_0.png)


With `wordcloud` and `matplotlib` libraries, we could create a simple wordcloud (without any customizations) just to have a feeling of what answers were generally included in the responses.

Here we could design a numbered value scheme and quickly code `male` and `female` into `1` and `2` respectively. We could researve `0` for `non-binary`.


```python
df['gender_r'] = 0
df.loc[df['gender'] == 'male', ['gender_r']] = 1
df.loc[df['gender'] == 'female', ['gender_r']] = 2
```


```python
df['gender_r'].value_counts().sort_index()
```




    0    325
    1    860
    2    248
    Name: gender_r, dtype: int64



Looks like we have 325 text answers that need to be further checked. I'm interested in what these answers look like and I would print out the top 5 most frequent descriptions of gender:


```python
df.loc[df['gender_r']==0, ['gender']]['gender'].value_counts()[0:5]
```




    m              165
    f               61
    woman            7
    man              5
    non-binary       4
    Name: gender, dtype: int64




```python
df.loc[df['gender'] == 'm', ['gender_r']] = 1
df.loc[df['gender'] == 'f', ['gender_r']] = 2
df.loc[df['gender'] == 'woman', ['gender_r']] = 1
df.loc[df['gender'] == 'man', ['gender_r']] = 2
```


```python
df.loc[df['gender_r']==0, ['gender']]['gender'].value_counts()[0:10]
```




    non-binary              4
    cis male                3
    nonbinary               2
    genderqueer             2
    human                   2
    agender                 2
    male (cis)              2
    genderflux demi-girl    1
    fluid                   1
    fm                      1
    Name: gender, dtype: int64



Frequency output shows that the texts haven't been fully cleaned yet. For example, "cis male" and "male (cis)" could be categorized into male.

#### Search for substring

We could run a simple search for substring to determine whether it should be categorized to either gender category or not.


```python
# exclude cleaned text answers
uncleaned_textdf = df.loc[df['gender_r']==0, ['ID', 'gender']]
```

The use of `in` is pretty simple - after search substring within string it will return a boolean value (`True` or `False`). See an example below:


```python
'male' in 'male (cis)'
```




    True




```python
# create a simple function that process data in batch
def checksubstring(substring, listofstring):
    booleanlist=[]
    for text in listofstring:
        booleanlist.append(substring in text)
    boolean_dic = {'text': listofstring,
                   'check': booleanlist}
    boolean_df = pd.DataFrame(boolean_dic)
    return boolean_df
```


```python
checksubstring('female', uncleaned_textdf['gender'].dropna())[0:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>check</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>i identify as female.</td>
      <td>True</td>
    </tr>
    <tr>
      <th>29</th>
      <td>bigender</td>
      <td>False</td>
    </tr>
    <tr>
      <th>30</th>
      <td>non-binary</td>
      <td>False</td>
    </tr>
    <tr>
      <th>42</th>
      <td>female assigned at birth</td>
      <td>True</td>
    </tr>
    <tr>
      <th>103</th>
      <td>fm</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



You probably noticed that this isn't the best tool to use as simply by searching "male" and assigning it back to the "male" category, we may mis-catgorize those that for example mentioned "male/genderqueer". For only around 60 uncategorized texts, it's probably easier to just manually review and code them to the appropriate categories. Sometimes, we/human could work with computer to acheive the best performance and highest efficiency. 

Another way to deal with texts is to use word embeddings from the very beginning. I'll illustrate this method a bit more in the upcoming posts.
