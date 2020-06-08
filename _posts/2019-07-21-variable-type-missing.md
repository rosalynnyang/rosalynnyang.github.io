---
title: "Checking survey data (2): change variable type and inspect missingness"
date: 2019-07-05
header:
  image: /assets/images/post2/background.jpg
  teaser: /assets/images/post2/background.jpg
  caption: image from web
tags:
- survey
- pandas
- numpy
- missing data
- variable type
categories:
- survey
---

In my [last post](https://rosalynnyang.github.io/survey/checking-survey-data/) in this particular series, I discussed the few steps that I usually take when first get in touch with the survey data. There were a few things outstanding, including mis-categorized variable types and inconsistent missing values. <br>
In this post, I'll address these two problems when further checking the data.


```python
import numpy as np
import pandas as pd
```


```python
df = pd.read_parquet('../output/df')
```

#### Change variable type

List all the columns in the dataset.


```python
df.columns
```




    Index(['self', 'employee', 'type', 'IT', 'benefit', 'options', 'discussion',
           'learn', 'anoymity', 'leave', 'mental_employer', 'physical_employer',
           'mental_coworker', 'mental_super', 'as_serious', 'consequence',
           'coverage', 'resources', 'reveal_client', 'impact', 'reveal_coworker',
           'impact_neg', 'productivity', 'time_affected', 'ex', 'coverage_ex',
           'options_ex', 'discussion_ex', 'learn_ex', 'anoymity_ex',
           'mental_employer_ex', 'physical_employer_ex', 'mental_coworker_ex',
           'mental_super_ex', 'as_serious_ex', 'consequence_ex',
           'physical_interview', 'physical_interview_y', 'mental_interview',
           'mental_interview_y', 'hurt', 'negative', 'share', 'unsupport',
           'experience', 'history', 'past', 'mental_disorder',
           'conditions_diagosed1', 'conditions_believe', 'diagnose',
           'conditions_diagosed2', 'treatment', 'interfere',
           'interfere_nottreated', 'age', 'gender', 'country_live', 'state',
           'country_work', 'state_work', 'position', 'remote', 'ID'],
          dtype='object')



Using `df.dtypes` and `df['variable'].unique()` in the last post, we could confirm the type of the variable and the unique levels/values of those variables. For example, the "whether have any mental health coverage" question below, data type shows it's a float type but it takes only on values 1, 0 and nan=missing value, which should be edited into a categorical variable. <br>
Note althought it didn't display as we intended, this is not neccessarily an error - Python just considers `nan` as a float. 


```python
df['coverage'].dtypes
```




    dtype('float64')




```python
df['coverage'].unique()
```




    array([nan,  1.,  0.])



What about checking all variables in the dataset at the same time?


```python
df.dtypes[0:10]
```




    self            int64
    employee       object
    type          float64
    IT            float64
    benefit        object
    options        object
    discussion     object
    learn          object
    anoymity       object
    leave          object
    dtype: object




```python
df_variables_list = df.columns.to_list()
variable_values_list = [df[col].unique() for col in df_variables_list] # using a list comprehension here instead of for loop
variable_values_list[0:10]
```




    [array([0, 1], dtype=int64),
     array(['26-100', '6-25', None, 'More than 1000', '100-500', '500-1000',
            '1-5'], dtype=object),
     array([ 1., nan,  0.]),
     array([nan,  1.,  0.]),
     array(['Not eligible for coverage / N/A', 'No', None, 'Yes',
            "I don't know"], dtype=object),
     array([None, 'Yes', 'I am not sure', 'No'], dtype=object),
     array(['No', 'Yes', None, "I don't know"], dtype=object),
     array(['No', 'Yes', None, "I don't know"], dtype=object),
     array(["I don't know", 'Yes', None, 'No'], dtype=object),
     array(['Very easy', 'Somewhat easy', 'Neither easy nor difficult', None,
            'Very difficult', 'Somewhat difficult', "I don't know"],
           dtype=object)]



After checking data types and values for all variables, we could decide which variables should actually be categorical vs. numerical vs. text. In Python, these are either integer, float, or string types. For survey data, string type is usually reserved for open-ended question texts. We can use `.astype('type')` to modify variable types.


```python
# converting variable types
shouldbeInt = ['type', 'IT', 'coverage']
for col in shouldbeInt:
    df[col] = df[col].astype('Int64') # note capitalized Int64 instead of int64 here - Pandas 0.24+ supports Int64 type with nan value
```

Check whether one of the variables have been converted correctly:


```python
df['type'].value_counts(dropna=False).sort_index()
```




    0      263
    1      883
    NaN    287
    Name: type, dtype: int64



Some variables just have response options texts recorded as their values, such as this scale question below that asks "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be":


```python
df['leave'].unique()
```




    array(['Very easy', 'Somewhat easy', 'Neither easy nor difficult', None,
           'Very difficult', 'Somewhat difficult', "I don't know"],
          dtype=object)



The `object` type suggests that it is a string variable. It is true for now as the values are texts only, but we would want it to be a categorical integer variable via recoding:


```python
# create a new variable for the recode so we could compare with the original; this can be omitted if familiar with recoding
df['leave_r'] = 0 # value 0 entails both "i don't know" and NaN values
df.loc[df['leave'] == 'Very easy', ['leave_r']] = 1
df.loc[df['leave'] == 'Somewhat easy', ['leave_r']] = 2
df.loc[df['leave'] == 'Neither easy nor difficult', ['leave_r']] = 3
df.loc[df['leave'] == 'Somewhat difficult', ['leave_r']] = 4
df.loc[df['leave'] == 'Very difficult', ['leave_r']] = 5
```

Compare the newly recoded variable with the original one:


```python
df['leave_r'].value_counts().sort_index()
```




    0    437
    1    220
    2    281
    3    178
    4    199
    5    118
    Name: leave_r, dtype: int64




```python
df['leave'].value_counts(dropna=False)
```




    NaN                           287
    Somewhat easy                 281
    Very easy                     220
    Somewhat difficult            199
    Neither easy nor difficult    178
    I don't know                  150
    Very difficult                118
    Name: leave, dtype: int64



#### Different missingness

`leave_r` is a quick and "dirty" cleaned up version of the original string variable. It's worth noting that in `leave_r` the value 0 entails both original `NaN` values and those who answered `I don't know` to this questions. There is a difference. <br>
In the last post, I illustrated with a bar plot that more than one variable has 287 missing values including this `leave` variable. Looking at the list of questions we could infer a logic that respondents who indicated they were self-employeed (variable `self`) in the first question were not asked this question. <br>
In other words, we could distinguish between intentional skips (I don't know) and programming skips (e.g. if Q1=2, skip to Q3).


```python
df['self'].value_counts()
```




    0    1146
    1     287
    Name: self, dtype: int64



How to proceed? <br>
We could code the programming skips to `-1` so when looking at the frequencies of certain variables we know these number of respondents were not eligible for the questions. For some other analyses, we could just ignore this difference by using `self` to subset data (e.g. self=0 corresponds to sum(leave) = 1146).
