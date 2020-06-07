---
title: "Checking Survey Data"
date: 2019-07-05
header:
  image: /assets/images/checking/background.jpeg
  teaser: /assets/images/checking/background.jpeg
  #caption: Illustration of Google Map Platform
tags:
- survey
- pandas
- numpy
- missing data
categories:
- Survey
---

## Examine and clean survey data

In this post, I'll discuss the usual steps I take when first checking survey data. This post uses data from 2016 OSMI Mental Health in Tech Survey, which is made publicly available at https://www.kaggle.com/osmi/mental-health-in-tech-2016


```python
# load numpy, pandas for data manipulation and matplotlib for quick visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
# import csv data
df = pd.read_csv('../input/IT_mental_health.csv')
```


```python
# examine the shape of the data
df.shape
```




    (1433, 63)



The shape of the dataset indicates that there were n=1433 respondents who participated in the survey and 63 columns of information were collected. If this data has not been cleaned before, there could be a chance that there might be duplicated records on the rows and less-informational columns that we may want to reconsider. The next steps involve checking the data on both the rows (records/respondents) and columns (variables/features) aspects.


```python
# take a look at the first few columns
df.head(3)
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
      <th>self</th>
      <th>employee</th>
      <th>type</th>
      <th>IT</th>
      <th>benefit</th>
      <th>options</th>
      <th>discussion</th>
      <th>learn</th>
      <th>anoymity</th>
      <th>leave</th>
      <th>...</th>
      <th>age</th>
      <th>gender</th>
      <th>country_live</th>
      <th>state</th>
      <th>country_work</th>
      <th>state_work</th>
      <th>position</th>
      <th>remote</th>
      <th>id</th>
      <th>ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>26-100</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>Not eligible for coverage / N/A</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>I don't know</td>
      <td>Very easy</td>
      <td>...</td>
      <td>39</td>
      <td>Male</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>Back-end Developer</td>
      <td>Sometimes</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>6-25</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Somewhat easy</td>
      <td>...</td>
      <td>29</td>
      <td>male</td>
      <td>United States of America</td>
      <td>Illinois</td>
      <td>United States of America</td>
      <td>Illinois</td>
      <td>Back-end Developer|Front-end Developer</td>
      <td>Never</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>6-25</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>No</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>I don't know</td>
      <td>Neither easy nor difficult</td>
      <td>...</td>
      <td>38</td>
      <td>Male</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>United Kingdom</td>
      <td>NaN</td>
      <td>Back-end Developer</td>
      <td>Always</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 65 columns</p>
</div>



The first impression looking at the head of the dataset is that the columns have the full question texts as their names. Let's first check whether that's the case:


```python
# check current column names and see what the variables are
question_list = df.columns.to_list()
len(question_list)
```




    65




```python
question_list[0:5]
```




    ['Are you self-employed?',
     'How many employees does your company or organization have?',
     'Is your employer primarily a tech company/organization?',
     'Is your primary role within your company related to tech/IT?',
     'Does your employer provide mental health benefits as part of healthcare coverage?']



By checking {df.columns} we know that indeed the data columns are named with their original question texts. We usually would want the variable names to be concise yet meaningful. If possible, we could rename by assigning short phrases as column names, or just simply use Q1, Q2, ...Q63 as names.


```python
# assign variable names
df.columns = ["self", "employee", "type", "IT", "benefit", "options", "discussion", "learn", "anoymity", "leave", "mental_employer", "physical_employer",
              "mental_coworker", "mental_super", "as_serious", "consequence", "coverage", "resources", "reveal_client", "impact", "reveal_coworker",
              "impact_neg", "productivity", "time_affected", "ex", "coverage_ex", "options_ex", "discussion_ex", "learn_ex", "anoymity_ex", "mental_employer_ex",
              "physical_employer_ex", "mental_coworker_ex", "mental_super_ex", "as_serious_ex", "consequence_ex", "physical_interview", "physical_interview_y", 
              "mental_interview", "mental_interview_y", "hurt", "negative", "share", "unsupport", "experience", "history", "past", "mental_disorder",
              "conditions_diagosed1", "conditions_believe", "diagnose", "conditions_diagosed2", "treatment", "interfere", "interfere_nottreated",
              "age", "gender", "country_live", "state", "country_work", "state_work", "position", "remote"]
```

Another thing about the dataset is that it does not have a unique identifier, meaning that there is not a respondentID. We could simply use index of the rows to be respondent ID.


```python
# use index as IDs
df["ID"] = df.index + 1
```

Next, let's look at the rows. There are a few things we could check: 1) duplicated records (respondents entered exactly same values across all survey questions, this is likely an error since there are 63 questions in the study; 2) missing data (some due to legitimate skips, some are truly skips, and others may be programming errors, and we need to distinguish between these missingness)


```python
# check for duplicated answers
df.duplicated().sum()
```




    0




```python
# create a function for summarizing n, number of missing values, variable types, and values of each variable/column
def summarize_variable(data):
    valuelist =[]
    for col in data.columns:
        valuelist.append(data[col].unique())
    summarydic = {'variable_name': data.columns, 
                  'n': data.notnull().sum().to_list(),
                  'n_miss': data.isnull().sum().to_list(),
                  'values': valuelist,
                  'variable_type': data.dtypes}
    summarydf = pd.DataFrame(summarydic)
    return summarydf
```


```python
summarytable = summarize_variable(df)
summarytable
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
      <th>variable_name</th>
      <th>n</th>
      <th>n_miss</th>
      <th>values</th>
      <th>variable_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>self</th>
      <td>self</td>
      <td>1433</td>
      <td>0</td>
      <td>[0, 1]</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>employee</th>
      <td>employee</td>
      <td>1146</td>
      <td>287</td>
      <td>[26-100, 6-25, nan, More than 1000, 100-500, 5...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>type</th>
      <td>type</td>
      <td>1146</td>
      <td>287</td>
      <td>[1.0, nan, 0.0]</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>IT</th>
      <td>IT</td>
      <td>263</td>
      <td>1170</td>
      <td>[nan, 1.0, 0.0]</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>benefit</th>
      <td>benefit</td>
      <td>1146</td>
      <td>287</td>
      <td>[Not eligible for coverage / N/A, No, nan, Yes...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>options</th>
      <td>options</td>
      <td>1013</td>
      <td>420</td>
      <td>[nan, Yes, I am not sure, No]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>discussion</th>
      <td>discussion</td>
      <td>1146</td>
      <td>287</td>
      <td>[No, Yes, nan, I don't know]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>learn</th>
      <td>learn</td>
      <td>1146</td>
      <td>287</td>
      <td>[No, Yes, nan, I don't know]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>anoymity</th>
      <td>anoymity</td>
      <td>1146</td>
      <td>287</td>
      <td>[I don't know, Yes, nan, No]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>leave</th>
      <td>leave</td>
      <td>1146</td>
      <td>287</td>
      <td>[Very easy, Somewhat easy, Neither easy nor di...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>mental_employer</th>
      <td>mental_employer</td>
      <td>1146</td>
      <td>287</td>
      <td>[No, Maybe, nan, Yes]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>physical_employer</th>
      <td>physical_employer</td>
      <td>1146</td>
      <td>287</td>
      <td>[No, nan, Maybe, Yes]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>mental_coworker</th>
      <td>mental_coworker</td>
      <td>1146</td>
      <td>287</td>
      <td>[Maybe, nan, Yes, No]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>mental_super</th>
      <td>mental_super</td>
      <td>1146</td>
      <td>287</td>
      <td>[Yes, Maybe, nan, No]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>as_serious</th>
      <td>as_serious</td>
      <td>1146</td>
      <td>287</td>
      <td>[I don't know, Yes, nan, No]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>consequence</th>
      <td>consequence</td>
      <td>1146</td>
      <td>287</td>
      <td>[No, nan, Yes]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>coverage</th>
      <td>coverage</td>
      <td>287</td>
      <td>1146</td>
      <td>[nan, 1.0, 0.0]</td>
      <td>float64</td>
    </tr>
    <tr>
      <th>resources</th>
      <td>resources</td>
      <td>287</td>
      <td>1146</td>
      <td>[nan, Yes, I know several, I know some, No, I ...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>reveal_client</th>
      <td>reveal_client</td>
      <td>287</td>
      <td>1146</td>
      <td>[nan, Sometimes, if it comes up, No, because i...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>impact</th>
      <td>impact</td>
      <td>144</td>
      <td>1289</td>
      <td>[nan, I'm not sure, Yes, No]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>reveal_coworker</th>
      <td>reveal_coworker</td>
      <td>287</td>
      <td>1146</td>
      <td>[nan, Sometimes, if it comes up, No, because i...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>impact_neg</th>
      <td>impact_neg</td>
      <td>287</td>
      <td>1146</td>
      <td>[nan, I'm not sure, No, Yes, Not applicable to...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>productivity</th>
      <td>productivity</td>
      <td>287</td>
      <td>1146</td>
      <td>[nan, Yes, Not applicable to me, No, Unsure]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>time_affected</th>
      <td>time_affected</td>
      <td>204</td>
      <td>1229</td>
      <td>[nan, 1-25%, 76-100%, 26-50%, 51-75%]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>ex</th>
      <td>ex</td>
      <td>1433</td>
      <td>0</td>
      <td>[1, 0]</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>coverage_ex</th>
      <td>coverage_ex</td>
      <td>1264</td>
      <td>169</td>
      <td>[No, none did, Yes, they all did, Some did, I ...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>options_ex</th>
      <td>options_ex</td>
      <td>1264</td>
      <td>169</td>
      <td>[N/A (not currently aware), I was aware of som...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>discussion_ex</th>
      <td>discussion_ex</td>
      <td>1264</td>
      <td>169</td>
      <td>[I don't know, None did, Some did, nan, Yes, t...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>learn_ex</th>
      <td>learn_ex</td>
      <td>1264</td>
      <td>169</td>
      <td>[None did, Some did, nan, Yes, they all did]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>anoymity_ex</th>
      <td>anoymity_ex</td>
      <td>1264</td>
      <td>169</td>
      <td>[I don't know, Yes, always, Sometimes, No, nan]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>consequence_ex</th>
      <td>consequence_ex</td>
      <td>1264</td>
      <td>169</td>
      <td>[None of them, Some of them, nan, Yes, all of ...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>physical_interview</th>
      <td>physical_interview</td>
      <td>1433</td>
      <td>0</td>
      <td>[Maybe, Yes, No]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>physical_interview_y</th>
      <td>physical_interview_y</td>
      <td>1095</td>
      <td>338</td>
      <td>[nan, It would depend on the health issue. If ...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>mental_interview</th>
      <td>mental_interview</td>
      <td>1433</td>
      <td>0</td>
      <td>[Maybe, No, Yes]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>mental_interview_y</th>
      <td>mental_interview_y</td>
      <td>1126</td>
      <td>307</td>
      <td>[nan, While mental health has become a more pr...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>hurt</th>
      <td>hurt</td>
      <td>1433</td>
      <td>0</td>
      <td>[Maybe, No, I don't think it would, Yes, I thi...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>negative</th>
      <td>negative</td>
      <td>1433</td>
      <td>0</td>
      <td>[No, I don't think they would, Maybe, Yes, the...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>share</th>
      <td>share</td>
      <td>1433</td>
      <td>0</td>
      <td>[Somewhat open, Neutral, Not applicable to me ...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>unsupport</th>
      <td>unsupport</td>
      <td>1344</td>
      <td>89</td>
      <td>[No, Maybe/Not sure, Yes, I experienced, Yes, ...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>experience</th>
      <td>experience</td>
      <td>657</td>
      <td>776</td>
      <td>[nan, Yes, No, Maybe]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>history</th>
      <td>history</td>
      <td>1433</td>
      <td>0</td>
      <td>[No, Yes, I don't know]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>past</th>
      <td>past</td>
      <td>1433</td>
      <td>0</td>
      <td>[Yes, Maybe, No]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>mental_disorder</th>
      <td>mental_disorder</td>
      <td>1433</td>
      <td>0</td>
      <td>[No, Yes, Maybe]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>conditions_diagosed1</th>
      <td>conditions_diagosed1</td>
      <td>568</td>
      <td>865</td>
      <td>[nan, Anxiety Disorder (Generalized, Social, P...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>conditions_believe</th>
      <td>conditions_believe</td>
      <td>322</td>
      <td>1111</td>
      <td>[nan, Substance Use Disorder|Addictive Disorde...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>diagnose</th>
      <td>diagnose</td>
      <td>1433</td>
      <td>0</td>
      <td>[Yes, No]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>conditions_diagosed2</th>
      <td>conditions_diagosed2</td>
      <td>711</td>
      <td>722</td>
      <td>[Anxiety Disorder (Generalized, Social, Phobia...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>treatment</th>
      <td>treatment</td>
      <td>1433</td>
      <td>0</td>
      <td>[0, 1]</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>interfere</th>
      <td>interfere</td>
      <td>1433</td>
      <td>0</td>
      <td>[Not applicable to me, Rarely, Sometimes, Neve...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>interfere_nottreated</th>
      <td>interfere_nottreated</td>
      <td>1433</td>
      <td>0</td>
      <td>[Not applicable to me, Sometimes, Often, Rarel...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>age</th>
      <td>age</td>
      <td>1433</td>
      <td>0</td>
      <td>[39, 29, 38, 43, 42, 30, 37, 44, 28, 34, 35, 5...</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>gender</th>
      <td>gender</td>
      <td>1430</td>
      <td>3</td>
      <td>[Male, male, Male , Female, M, female, m, I id...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>country_live</th>
      <td>country_live</td>
      <td>1433</td>
      <td>0</td>
      <td>[United Kingdom, United States of America, Can...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>state</th>
      <td>state</td>
      <td>840</td>
      <td>593</td>
      <td>[nan, Illinois, Tennessee, Virginia, Californi...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>country_work</th>
      <td>country_work</td>
      <td>1433</td>
      <td>0</td>
      <td>[United Kingdom, United States of America, Can...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>state_work</th>
      <td>state_work</td>
      <td>851</td>
      <td>582</td>
      <td>[nan, Illinois, Tennessee, Virginia, Californi...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>position</th>
      <td>position</td>
      <td>1433</td>
      <td>0</td>
      <td>[Back-end Developer, Back-end Developer|Front-...</td>
      <td>object</td>
    </tr>
    <tr>
      <th>remote</th>
      <td>remote</td>
      <td>1433</td>
      <td>0</td>
      <td>[Sometimes, Never, Always]</td>
      <td>object</td>
    </tr>
    <tr>
      <th>id</th>
      <td>id</td>
      <td>1433</td>
      <td>0</td>
      <td>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14...</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>ID</th>
      <td>ID</td>
      <td>1433</td>
      <td>0</td>
      <td>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14...</td>
      <td>int64</td>
    </tr>
  </tbody>
</table>
<p>65 rows × 5 columns</p>
</div>




```python
# create a visual showing the number of missing values summarized by variable
summarytable['n_miss'].value_counts()[1:].plot(kind='bar', figsize=(12,8))
plt.xlabel("number of missing", labelpad=10, fontsize=12)
plt.xticks(rotation=360)
plt.ylabel("Number of variables", labelpad=10, fontsize=12)
plt.title("Number of variables that have certain number of missing values", y=1.02, fontsize=15)
plt.show()
```


![png](/assets/images/checking/output_18_0.png)


From the bar plot above, we can see that there are a number of variables that have missing values at 287, 169, and 1146, which implies that the missingness could be due to legitimate skip of the survey questions. The rest of the missing values are only specific to certain variables, which could be due to respondent skipping that question (don't know or refusal). For now, we don't want to recode or impute for missing values. These procedures are usually carried out with specific analysis intentions.
