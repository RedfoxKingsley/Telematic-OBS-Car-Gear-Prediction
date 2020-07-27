

```python
#Setting It Up
#I collected all of the data above and combined them into one dataframe. The code and details are located here. One challenge was the periodicity of the various features. Our exchange data is daily, some data is monthly, and others quarterly. For our daily exchange rates, I took the last value of each month. For the quarterly data, I copied the quarterly value to each month in that quarter. This gives us a dataframe of monthly data that is easier to work with.
#First, we will import the libraries we will be using and also load our data into a Pandas dataframe.

# Import needed libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn as sn
import sklearn

# Python magic to show plots inline in the notebook
%matplotlib inline
plt.style.use('ggplot')
import datetime as dt
from datetime import datetime
import math
```


```python
# Import data
df = pd.read_csv("https://raw.githubusercontent.com/Preetinsights/Telematic-OBS-Gear-Prediction/master/allcars.csv")
```


```python
print(df)
```

            Unnamed: 0  Unnamed: 0.1 timeStamp  tripID  \
    0           117459           184   28:21.0       0   
    1           117460           185   28:22.0       0   
    2           117461           186   28:23.0       0   
    3           117462           187   28:24.0       0   
    4           117463           188   28:25.0       0   
    5           117464           189   28:26.0       0   
    6           117465           190   28:27.0       0   
    7           117466           191   28:28.0       0   
    8           117467           192   28:29.0       0   
    9           117468           193   28:30.0       0   
    10          117469           194   28:31.0       0   
    11          117470           195   28:32.0       0   
    12          117471           196   28:33.0       0   
    13          117472           197   28:34.0       0   
    14          117473           198   28:35.0       0   
    15          117474           199   28:36.0       0   
    16          117475           200   28:37.0       0   
    17          117476           201   28:38.0       0   
    18          117477           202   28:39.0       0   
    19          117478           203   28:40.0       0   
    20          117479           204   28:41.0       0   
    21          117480           205   28:42.0       0   
    22          117481           206   28:43.0       0   
    23          117482           207   28:44.0       0   
    24          117483           208   28:45.0       0   
    25          117484           209   28:46.0       0   
    26          117485           210   28:47.0       0   
    27          117486           211   28:48.0       0   
    28          117487           212   28:49.0       0   
    29          117488           213   28:50.0       0   
    ...            ...           ...       ...     ...   
    106430      223904        106629   22:36.0     126   
    106431      223905        106630   22:37.0     126   
    106432      223906        106631   22:38.0     126   
    106433      223907        106632   22:39.0     126   
    106434      223908        106633   22:39.5     126   
    106435      223909        106634   22:41.0     126   
    106436      223910        106635   22:42.0     126   
    106437      223911        106636   22:43.0     126   
    106438      223912        106637   22:44.0     126   
    106439      223913        106638   22:44.5     126   
    106440      223914        106639   22:46.0     126   
    106441      223915        106640   22:47.0     126   
    106442      223916        106641   22:48.0     126   
    106443      223917        106642   22:49.0     126   
    106444      223918        106643   22:50.0     126   
    106445      223919        106644   22:51.0     126   
    106446      223920        106645   22:52.0     126   
    106447      223921        106646   22:53.0     126   
    106448      223922        106647   22:54.0     126   
    106449      223923        106648   22:55.0     126   
    106450      223924        106649   22:56.0     126   
    106451      223925        106650   22:57.0     126   
    106452      223926        106651   22:58.0     126   
    106453      223927        106652   22:59.0     126   
    106454      223928        106653   23:00.0     126   
    106455      223929        106654   23:00.5     126   
    106456      223930        106655   23:01.0     126   
    106457      223931        106656   23:02.0     126   
    106458      223932        106657   23:04.0     126   
    106459      223933        106658   28:20.0     126   
    
                                                      accData  gps_speed  battery  \
    0       0f18fe2806d00210bf030fc1fe0ebffe0ec0fd10c0ff0e...        0.0      0.0   
    1       0f48fe400660fe0dc1ff0ebfff0fc0010ebefd0dc0010f...        0.0      0.0   
    2       0ef8fe300678fe0ebfff0ec0030fc0ff0dc1000fc0000e...        0.0      0.0   
    3       0f20fe2806d8ff0cc0ff0dc2000fc1ff0ec1010dbe000e...        2.4      0.0   
    4       0f50fe800678fe10c0000ec0000ec0000ebf0110c00010...        2.7      0.0   
    5       0f18fe480648fe0fc0000dbefd0cc2fe0ec3fe0fc2ff0d...        5.2      0.0   
    6       0ee0fe4006a0ff0ec0fe0ebdff0fc2fe0ec0ff0ec0ff0e...        8.2      0.0   
    7       0f10fe300698ff0dbffe0dbe010fc1000fc0ff0ec0fe0e...        0.0      0.0   
    8       0f58fe4006b8000fc0000ec1000fc2ff0ec2fe0dc0ff0e...        0.0      0.0   
    9       0f20fe2006d0ff0ec1ff0dbfff0dbd000ec1ff0dbf000e...        0.0      0.0   
    10      0ef8fe4006b8000ebffe0ec1000ec0ff0fc2ff0fc2000e...        0.0      0.0   
    11      0f38fe3006b0000dbfff0ec1ff0fbfff0dc1000fc0010d...        0.9      0.0   
    12      0f78fe2806a0000ebf000ebfff0ebefe0cc00210c10010...        0.9      0.0   
    13      0ef8fe000688fe0ec2fe0dc0000ebf000ec1ff0dc0000f...        0.0      0.0   
    14      0f30fe1006d0ff0ec0fd0ebf000fc0ff0dbfff0ec0000e...        0.0      0.0   
    15      0ef8fe4806a8000ec3000fc1fe0ebfff0fc0ff10c0ff0e...        0.0      0.0   
    16      0f18fe000690020ebfff0fc1fe0ec1ff0fc1ff0ec0ff0f...        0.0      0.0   
    17      0f48fe280678ff0ec0000ebfff0ebefe0fc0000dc0000e...        0.0      0.0   
    18      0f20fe300660000dbf010dc2000fbfff0ebfff0ec1ff0e...        0.0      0.0   
    19      0f28fe1806c0ff0ec1000ec0fe0ec0020ebcfe0cbefe0d...        0.0      0.0   
    20      0f18fe480698fe0dbe000ebfff0bbe010ec0ff0ec0000f...        2.2      0.0   
    21      0f30fe200650000ec0000ec0ff0dc0ff0dbeff0dc0000f...        4.4      0.0   
    22      0ef8fe300690ff0dc2fe0dc00010c1ff0dbe000ec0ff0e...        5.0      0.0   
    23      0f40fe480698000fc1000ec0ff0ec1ff0fc1ff0dc0000e...        4.1      0.0   
    24      0f50fe080658010ec1ff0dc0ff0dc0ff0fc0000ec0ff0d...        2.6      0.0   
    25      0f78fe300658000fc2fe0dbf000ebeff0ec0fe0ec0ff0e...        1.9      0.0   
    26      0f30fe180668ff0fc0000ec0000fc0ff0dbfff0dc0ff0d...        0.0      0.0   
    27      0f60fe300688ff0dbe000dc1000ec1000fc0000fbf000d...        0.0      0.0   
    28      0f40fe3806b0fd0ebf000ec0fe0fc0010fc0ff0ec1fe0e...        0.0      0.0   
    29      0f48fe4806a8000ebf000ec0000ec1ff0fbe0010c2fe0d...        0.0      0.0   
    ...                                                   ...        ...      ...   
    106430  0f00fe000680ff1abffc16c1010ec2fd0cc3fe13c1fc0d...        0.0      0.0   
    106431  0f30fe400658020dbfff0ec3fb11c4fb11c4030fc00210...        0.0      0.0   
    106432  0f20fe200668fa11bbf616bbfd0bc1fb14befa10bdfd09...        0.0      0.0   
    106433  0f10fe1806480113c1fc0bc4fe0bc0ff0cc2010cc0060e...        0.0      0.0   
    106434  0f18fde00638fb1db90206c2fc12bc000dbefd0ebf040e...        0.0      0.0   
    106435  0ed8fe2006c0020fc2040dc0fe0cc1fb0dc0fd10c70312...        0.0      0.0   
    106436  0f18fe200638fa10c0010fbcfe1cb7ff0ec1000cc5fe17...        0.0      0.0   
    106437  0f28fe280620f722b50504c7fb11bc000cc3050ec5020f...        0.0      0.0   
    106438  0f28fe2806500302c5f61ebbfc10bdfc10bd000fc00109...        0.0      0.0   
    106439  0ee8fdd00658ff12c0050bc2fd15bffa20bcff09cbff18...        0.0      0.0   
    106440  0ed8fe100648f91cc0fd0abc0210c50807c70014befc10...        0.0      0.0   
    106441  0f28fe280680fa19bdff09c0010ec1fa0fc3ff0fc1090e...        0.0      0.0   
    106442  0f00fe100660fc11c20214c0ff0cbe010dbefb11baff0c...        0.0      0.0   
    106443  0f08fe180670fd0ebdfe07c0fe18bbfd0ec0fd1bbbfb13...        0.0      0.0   
    106444  0f00fe3006180312befc1ac4fe0ebffd16bd0109c5fe10...        0.0      0.0   
    106445  0f08fde00670fdffc60101c80013bcfd10be040fbdfe0c...        0.0      0.0   
    106446  0ed0fe180640020fc2ff11c4030fbfff13bbfefcbef813...        0.0      0.0   
    106447  0f20fe200670050bb6080dc1f822b6ff14bbfa25b7faff...        0.0      0.0   
    106448  0ef8fdf006180008c408faca0306c7fe15b8fc16bb070a...        0.0      0.0   
    106449  0f30fde00648ff11bdff0dbd030dc4f408be030bba060b...        0.0      0.0   
    106450  0f28fe300690f50ec10805c0fa17b80401c8fa16bcfa10...        0.0      0.0   
    106451  0ee8fe000658090cbefd16bffd16bd0311c3010ac3fa0b...        0.0      0.0   
    106452  0ef0fe1006180012c6fe0dbe080dbbf611b90806bc0421...        0.0      0.0   
    106453  0f08fe280680050fc0f60abf0402c3ff18beff0fbd030a...        0.0      0.0   
    106454  0ef0fdf00670ff0bbe0313bef50ec2fe0ac0fd0dbefb13...        0.0      0.0   
    106455  0f18fe200608fe0ec1fe0cc7010abd0013bf010dc10410...        0.0      0.0   
    106456  0f10fe1006800111c1020bbefc17beff0cbffe11c0ff11...        0.0      0.0   
    106457  0ee8fe1006a8ff11c10611c4ff11c0000dc3f90fc80111...        0.0      0.0   
    106458  0f08fe0006580511c1fd10bcfb13c1fe16bd000bc10315...        0.0      0.0   
    106459  0f50fe7806d8030dbfff0ec00000000000000000000000...        0.0      0.0   
    
            cTemp  dtc  eLoad  iat  imap  kpl  maf  rpm  speed  tAdv  tPos  \
    0         0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    1         0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    2         0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    3         0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    4         0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    5         0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    6         0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    7         0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    8         0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    9         0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    10        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    11        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    12        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    13        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    14        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    15        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    16        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    17        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    18        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    19        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    20        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    21        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    22        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    23        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    24        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    25        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    26        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    27        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    28        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    29        0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    ...       ...  ...    ...  ...   ...  ...  ...  ...    ...   ...   ...   
    106430    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106431    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106432    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106433    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106434    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106435    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106436    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106437    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106438    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106439    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106440    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106441    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106442    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106443    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106444    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106445    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106446    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106447    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106448    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106449    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106450    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106451    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106452    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106453    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106454    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106455    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106456    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106457    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106458    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    106459    0.0  0.0    0.0  0.0   0.0  0.0  0.0  0.0    0.0   0.0   0.0   
    
            deviceID  
    0              2  
    1              2  
    2              2  
    3              2  
    4              2  
    5              2  
    6              2  
    7              2  
    8              2  
    9              2  
    10             2  
    11             2  
    12             2  
    13             2  
    14             2  
    15             2  
    16             2  
    17             2  
    18             2  
    19             2  
    20             2  
    21             2  
    22             2  
    23             2  
    24             2  
    25             2  
    26             2  
    27             2  
    28             2  
    29             2  
    ...          ...  
    106430         2  
    106431         2  
    106432         2  
    106433         2  
    106434         2  
    106435         2  
    106436         2  
    106437         2  
    106438         2  
    106439         2  
    106440         2  
    106441         2  
    106442         2  
    106443         2  
    106444         2  
    106445         2  
    106446         2  
    106447         2  
    106448         2  
    106449         2  
    106450         2  
    106451         2  
    106452         2  
    106453         2  
    106454         2  
    106455         2  
    106456         2  
    106457         2  
    106458         2  
    106459         2  
    
    [106460 rows x 19 columns]
    


```python
#It is a good practice to understand the data first and try to gather as many insights from it. 
#EDA is all about making sense of data in hand,before getting them dirty with it.

#1. Check for Missing Data
#2. Heatmap & Data Structure
#3. Correlations
#4. Uncover a parsimonious model, one which explains the data with a minimum number of predictor variables.
df.head()
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
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>timeStamp</th>
      <th>tripID</th>
      <th>accData</th>
      <th>gps_speed</th>
      <th>battery</th>
      <th>cTemp</th>
      <th>dtc</th>
      <th>eLoad</th>
      <th>iat</th>
      <th>imap</th>
      <th>kpl</th>
      <th>maf</th>
      <th>rpm</th>
      <th>speed</th>
      <th>tAdv</th>
      <th>tPos</th>
      <th>deviceID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>117459</td>
      <td>184</td>
      <td>28:21.0</td>
      <td>0</td>
      <td>0f18fe2806d00210bf030fc1fe0ebffe0ec0fd10c0ff0e...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>117460</td>
      <td>185</td>
      <td>28:22.0</td>
      <td>0</td>
      <td>0f48fe400660fe0dc1ff0ebfff0fc0010ebefd0dc0010f...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>117461</td>
      <td>186</td>
      <td>28:23.0</td>
      <td>0</td>
      <td>0ef8fe300678fe0ebfff0ec0030fc0ff0dc1000fc0000e...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>117462</td>
      <td>187</td>
      <td>28:24.0</td>
      <td>0</td>
      <td>0f20fe2806d8ff0cc0ff0dc2000fc1ff0ec1010dbe000e...</td>
      <td>2.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>117463</td>
      <td>188</td>
      <td>28:25.0</td>
      <td>0</td>
      <td>0f50fe800678fe10c0000ec0000ec0000ebf0110c00010...</td>
      <td>2.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 106460 entries, 0 to 106459
    Data columns (total 19 columns):
    Unnamed: 0      106460 non-null int64
    Unnamed: 0.1    106460 non-null int64
    timeStamp       106460 non-null object
    tripID          106460 non-null int64
    accData         106460 non-null object
    gps_speed       106460 non-null float64
    battery         106460 non-null float64
    cTemp           106460 non-null float64
    dtc             106460 non-null float64
    eLoad           106460 non-null float64
    iat             106460 non-null float64
    imap            106460 non-null float64
    kpl             106460 non-null float64
    maf             106460 non-null float64
    rpm             106460 non-null float64
    speed           106460 non-null float64
    tAdv            106460 non-null float64
    tPos            106460 non-null float64
    deviceID        106460 non-null int64
    dtypes: float64(13), int64(4), object(2)
    memory usage: 15.4+ MB
    


```python
#Check for Missing Data
df.isnull().values.any()
```




    True




```python
# Show rows where any cell has a NaN
df[df.isnull().any(axis=1)].shape
```




    (0, 19)




```python
#### Drop cells with NaN
df = df.dropna(axis=0,subset=['cTemp'])
df = df.dropna(axis=0,subset=['dtc'])
df = df.dropna(axis=0,subset=['iat'])
df = df.dropna(axis=0,subset=['imap'])
df = df.dropna(axis=0,subset=['tAdv'])
```


```python
# Show rows where any cell has a NaN
df[df.isnull().any(axis=1)].shape
```




    (0, 19)




```python
df.describe()
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
      <th>Unnamed: 0</th>
      <th>Unnamed: 0.1</th>
      <th>tripID</th>
      <th>gps_speed</th>
      <th>battery</th>
      <th>cTemp</th>
      <th>dtc</th>
      <th>eLoad</th>
      <th>iat</th>
      <th>imap</th>
      <th>kpl</th>
      <th>maf</th>
      <th>rpm</th>
      <th>speed</th>
      <th>tAdv</th>
      <th>tPos</th>
      <th>deviceID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>106460.000000</td>
      <td>106460.000000</td>
      <td>106460.000000</td>
      <td>106460.000000</td>
      <td>106460.0</td>
      <td>106460.000000</td>
      <td>106460.0</td>
      <td>106460.000000</td>
      <td>106460.000000</td>
      <td>106460.000000</td>
      <td>106460.0</td>
      <td>106460.0</td>
      <td>106460.000000</td>
      <td>106460.000000</td>
      <td>106460.0</td>
      <td>106460.0</td>
      <td>106460.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>170703.476893</td>
      <td>53428.476893</td>
      <td>63.697849</td>
      <td>18.222948</td>
      <td>0.0</td>
      <td>64.143575</td>
      <td>0.0</td>
      <td>35.477576</td>
      <td>31.122901</td>
      <td>96.442175</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1183.945900</td>
      <td>33.075089</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>30732.539133</td>
      <td>30732.539133</td>
      <td>38.719864</td>
      <td>18.727147</td>
      <td>0.0</td>
      <td>29.107386</td>
      <td>0.0</td>
      <td>22.502089</td>
      <td>15.797552</td>
      <td>47.344598</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>759.576518</td>
      <td>33.972104</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>117459.000000</td>
      <td>184.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>144088.750000</td>
      <td>26813.750000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>60.000000</td>
      <td>0.0</td>
      <td>22.352941</td>
      <td>24.000000</td>
      <td>97.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>800.750000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>170703.500000</td>
      <td>53428.500000</td>
      <td>60.000000</td>
      <td>13.600000</td>
      <td>0.0</td>
      <td>80.000000</td>
      <td>0.0</td>
      <td>38.823529</td>
      <td>34.000000</td>
      <td>99.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1167.750000</td>
      <td>25.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>197318.250000</td>
      <td>80043.250000</td>
      <td>99.000000</td>
      <td>27.800000</td>
      <td>0.0</td>
      <td>81.000000</td>
      <td>0.0</td>
      <td>48.235294</td>
      <td>43.000000</td>
      <td>110.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1733.500000</td>
      <td>50.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>223933.000000</td>
      <td>106658.000000</td>
      <td>126.000000</td>
      <td>82.100000</td>
      <td>0.0</td>
      <td>84.000000</td>
      <td>0.0</td>
      <td>94.901961</td>
      <td>58.000000</td>
      <td>221.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3566.000000</td>
      <td>149.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Seaborn doesn't handle NaN values, so we can fill them with 0 for now.
df = df.fillna(value=0)
# Pair grid of key variables.
g = sns.PairGrid(df)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Pairwise Grid of Numeric Features');
```


![png](output_10_0.png)



```python
g = sns.PairGrid(df, vars=["gps_speed", "speed"])
g = g.map(plt.scatter)
```


![png](output_11_0.png)



```python
g = sns.PairGrid(df, vars=["iat", "rpm"])
g = g.map(plt.scatter)
#iat is in-board automatic transmission
#rpm = revolution per minute
```


![png](output_12_0.png)



```python
#To use linear regression for modelling,its necessary to remove correlated variables to improve your model.
#One can find correlations using pandas “.corr()” function and can visualize the correlation matrix using a heatmap in seaborn.

corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, cmap='Blues')
plt.title('Correlation Heatmap of Numeric Features')
```




    Text(0.5,1,'Correlation Heatmap of Numeric Features')




![png](output_13_1.png)



```python
#Select variables with complete dataset (no nan or zero)
df1 = pd.DataFrame(df,columns=['tripID','gps_speed','cTemp','eLoad','iat','imap','rpm','speed'])
```


```python
#Remove correlated variables before feature selection.
corrMatrix = df1.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

#Here, it can be infered that IAT – In-dash automatic transmission “iat” has strong positive correlation with circular temperature “cTemp”
```


![png](output_15_0.png)



```python
#Make final dataset
df.columns
```




    Index(['Unnamed: 0', 'Unnamed: 0.1', 'timeStamp', 'tripID', 'accData',
           'gps_speed', 'battery', 'cTemp', 'dtc', 'eLoad', 'iat', 'imap', 'kpl',
           'maf', 'rpm', 'speed', 'tAdv', 'tPos', 'deviceID'],
          dtype='object')




```python
cols = df.columns.tolist()
```


```python
df1.to_csv('allcars.csv')
```


```python
df1.dtypes
```




    tripID         int64
    gps_speed    float64
    cTemp        float64
    eLoad        float64
    iat          float64
    imap         float64
    rpm          float64
    speed        float64
    dtype: object




```python
# FEATURE ENGINEERING
# Define custom function to create lag values
def feature_lag(features):
    for feature in features:
        df[feature + '-lag1'] = df[feature].shift(1)
        df[feature + '-lag2'] = df[feature].shift(2)
        df[feature + '-lag3'] = df[feature].shift(3)
        df[feature + '-lag4'] = df[feature].shift(4)

# Define columns to create lags for
features = ['tripID','gps_speed','cTemp','eLoad','iat','imap','rpm','speed']

# Call custom function
feature_lag(features)
```


```python
#predict gps speed 3, 6, and 12 months ahead.

df1['y3'] = df.gps_speed.shift(-3)
df1['y6'] = df.gps_speed.shift(-6)
df1['y12'] = df.gps_speed.shift(-12)
```


```python
df1 = df1.dropna(axis=0,subset=['y3'])
df1 = df1.dropna(axis=0,subset=['y6'])
df1 = df1.dropna(axis=0,subset=['y12'])
```


```python
df1.dtypes
```




    tripID         int64
    gps_speed    float64
    cTemp        float64
    eLoad        float64
    iat          float64
    imap         float64
    rpm          float64
    speed        float64
    y3           float64
    y6           float64
    y12          float64
    dtype: object




```python
#Split into Training and Test Data
#Cross validation is always desired when training machine learning models to be able to trust the generality of the model created. We will split our data into training and test data using Scikit learn's built in tools. Also for scikit learn we need to separate our dataset into inputs and the feature being predicted (or X's and y's).

y = df1['gps_speed']
```


```python
X = df1.drop(['gps_speed'], axis=1)
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1234)
```


```python
X_train.shape, y_train.shape
```




    ((74513, 10), (74513,))




```python
X_test.shape, y_test.shape
```




    ((31935, 10), (31935,))




```python
X.columns
```




    Index(['tripID', 'cTemp', 'eLoad', 'iat', 'imap', 'rpm', 'speed', 'y3', 'y6',
           'y12'],
          dtype='object')




```python
df1.dtypes
```




    tripID         int64
    gps_speed    float64
    cTemp        float64
    eLoad        float64
    iat          float64
    imap         float64
    rpm          float64
    speed        float64
    y3           float64
    y6           float64
    y12          float64
    dtype: object




```python
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
# Create linear regression object
regr = LinearRegression()
```


```python
# Train the model using the training sets
regr.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
# Make predictions using the testing set
lin_pred = regr.predict(X_test)
```


```python
linear_regression_score = regr.score(X_test, y_test)
linear_regression_score
```




    0.9925489189517546




```python
linear_regression_score = regr.score(X_train, y_train)
linear_regression_score
```




    0.9923194367832849




```python
from math import sqrt
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, lin_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, lin_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, lin_pred))
```

    Coefficients: 
     [-4.80717894e-04  4.17246956e-02 -1.85684512e-02 -9.45400424e-03
     -1.56901951e-02 -1.03784532e-03  5.86184055e-01  5.43109400e-02
     -1.18445019e-01  3.86888018e-02]
    Root mean squared error: 1.62
    Mean absolute error: 0.99
    R-squared: 0.99
    


```python
plt.scatter(y_test, lin_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Linear Regression Predicted vs Actual')
plt.show()
```


![png](output_37_0.png)



```python
### Neural Network Regression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create MLPRegressor object
mlp = MLPRegressor()
```


```python
# Train the model using the training sets
mlp.fit(X_train, y_train)
```




    MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(100,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)




```python
# Score the model
neural_network_regression_score = mlp.score(X_test, y_test)
neural_network_regression_score
```




    0.9923027194668101




```python
# Score the model
neural_network_regression_score = mlp.score(X_train, y_train)
neural_network_regression_score
```




    0.9921729324306336




```python
# Make predictions using the testing set
nnr_pred = mlp.predict(X_test)
```


```python
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, nnr_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, nnr_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, nnr_pred))
```

    Root mean squared error: 1.65
    Mean absolute error: 1.00
    R-squared: 0.99
    


```python
plt.scatter(y_test, nnr_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Neural Network Regression Predicted vs Actual')
plt.show()
```


![png](output_44_0.png)



```python
###Lasso
from sklearn.linear_model import Lasso

lasso = Lasso()
```


```python
lasso.fit(X_train, y_train)
```




    Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)




```python
# Score the model
lasso_score = lasso.score(X_test, y_test)
lasso_score
```




    0.9923086310446717




```python
# Score the model
lasso_score = lasso.score(X_train, y_train)
lasso_score
```




    0.9920410049477479




```python
# Make predictions using the testing set
lasso_pred = lasso.predict(X_test)
```


```python
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, lasso_pred)))

# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, lasso_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, lasso_pred))
```

    Root mean squared error: 1.65
    Mean absolute error: 0.99
    R-squared: 0.99
    


```python
plt.scatter(y_test, lasso_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Lasso Predicted vs Actual')
plt.show()
```


![png](output_51_0.png)



```python
##ElasticNet
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
elasticnet.fit(X_train, y_train)
```




    ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
          max_iter=1000, normalize=False, positive=False, precompute=False,
          random_state=None, selection='cyclic', tol=0.0001, warm_start=False)




```python
elasticnet_score = elasticnet.score(X_test, y_test)
elasticnet_score
```




    0.9923404691889874




```python
elasticnet_score = elasticnet.score(X_test, y_test)
elasticnet_score
```




    0.9923404691889874




```python
elasticnet_pred = elasticnet.predict(X_test)
```


```python
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, elasticnet_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, elasticnet_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, elasticnet_pred))
```

    Root mean squared error: 1.64
    Mean absolute error: 0.99
    R-squared: 0.99
    


```python
###Decision Forest Regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create Random Forrest Regressor object
regr_rf = RandomForestRegressor(n_estimators=200, random_state=1234)
```

    C:\Users\Kingsley\Anaconda3\lib\site-packages\sklearn\ensemble\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d
    


```python
# Train the model using the training sets
regr_rf.fit(X_train, y_train)

```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
               oob_score=False, random_state=1234, verbose=0, warm_start=False)




```python
regr_rf.fit(X_test, y_test)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
               oob_score=False, random_state=1234, verbose=0, warm_start=False)




```python
# Score the model
decision_forest_score = regr_rf.score(X_test, y_test)
decision_forest_score
```




    0.9993148789215262




```python
# Make predictions using the testing set
regr_rf_pred = regr_rf.predict(X_test)
```


```python
from math import sqrt
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, regr_rf_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, regr_rf_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, regr_rf_pred))
```

    Root mean squared error: 0.49
    Mean absolute error: 0.27
    R-squared: 1.00
    


```python
features = X.columns
importances = regr_rf.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()
```


![png](output_63_0.png)



```python
plt.scatter(y_test, regr_rf_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Decision Forest Predicted vs Actual')
plt.show()
```


![png](output_64_0.png)



```python
#Extra Trees Regression

from sklearn.ensemble import ExtraTreesRegressor

extra_tree = ExtraTreesRegressor(n_estimators=200, random_state=1234)
```


```python
extra_tree.fit(X_train, y_train)
```




    ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None,
              max_features='auto', max_leaf_nodes=None,
              min_impurity_decrease=0.0, min_impurity_split=None,
              min_samples_leaf=1, min_samples_split=2,
              min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
              oob_score=False, random_state=1234, verbose=0, warm_start=False)




```python
extratree_score = extra_tree.score(X_test, y_test)
extratree_score
```




    0.9953993058895532




```python
extratree_score = extra_tree.score(X_train, y_train)
extratree_score
```




    0.9999897264247072




```python
extratree_pred = extra_tree.predict(X_test)
```


```python
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, extratree_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, extratree_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, extratree_pred))
```

    Root mean squared error: 1.27
    Mean absolute error: 0.70
    R-squared: 1.00
    


```python
features = X.columns
importances = extra_tree.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()
```


![png](output_71_0.png)



```python
plt.scatter(y_test, extratree_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Extra Trees Predicted vs Actual')
plt.show()
```


![png](output_72_0.png)



```python
#Evaluate Models
print("Scores:")
print("Linear regression score: ", linear_regression_score)
print("Neural network regression score: ", neural_network_regression_score)
print("Lasso regression score: ", lasso_score)
print("ElasticNet regression score: ", elasticnet_score)
print("Decision forest score: ", decision_forest_score)
print("Extra Trees score: ", extratree_score)
print("\n")
print("RMSE:")
print("Linear regression RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, lin_pred)))
print("Neural network RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, nnr_pred)))
print("Lasso RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, lasso_pred)))
print("ElasticNet RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, elasticnet_pred)))
print("Decision forest RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, regr_rf_pred)))
print("Extra Trees RMSE: %.2f"
      % sqrt(mean_squared_error(y_test, extratree_pred)))
```

    Scores:
    Linear regression score:  0.9923194367832849
    Neural network regression score:  0.9921729324306336
    Lasso regression score:  0.9920410049477479
    ElasticNet regression score:  0.9923404691889874
    Decision forest score:  0.9993148789215262
    Extra Trees score:  0.9999897264247072
    
    
    RMSE:
    Linear regression RMSE: 1.62
    Neural network RMSE: 1.65
    Lasso RMSE: 1.65
    ElasticNet RMSE: 1.64
    Decision forest RMSE: 0.49
    Extra Trees RMSE: 1.27
    


```python
from IPython.display import Image
Image(filename='mmv.png')
```




![png](output_74_0.png)




```python
import pandas as pd
"""
A framework script that tags the data points according to the gear and assigns it a color and plots the data. 
The gear detection is done by assuming the borders generated using any of the algorithms and placed in
the borders array. 
"""

%matplotlib notebook
import matplotlib.pyplot as plt


def get_gear(entry, borders):
    if entry['rpm'] == 0:
        return 0
    rat = entry['speed'] / entry['rpm'] * 1000
    if np.isnan(rat) or np.isinf(rat):
        return 0
    for i in range(0, len(borders)):
        if rat < borders[i] :
            return i + 1
    return 0

num_trips = 10
df = pd.read_csv("C:/Users/Kingsley/Desktop/allcars.csv", index_col=0)
obddata = df[df['tripID']<num_trips]

# borders = get_segment_borders(obddata)
borders = [7.070124715964856, 13.362448319790191, 19.945056624926686, 27.367647318253834, 32.17327586520911]

obddata_wgears = obddata
obddata_wgears['gear'] = obddata.apply(lambda x : get_gear(x, borders), axis=1)

# print(obddata_wgears)

colors = [x * 50 for x in obddata_wgears['gear']]
plt.scatter(obddata_wgears['rpm'], obddata_wgears['speed'], c=colors)
plt.plot()
```

    C:\Users\Kingsley\Anaconda3\lib\site-packages\ipykernel_launcher.py:31: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    


    <IPython.core.display.Javascript object>



<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAhwAAAFoCAYAAAAcpSI2AAAgAElEQVR4XuydB3RURRfH/1vSeyOE3jsICAqfCAIqIKIgIChKEVQEadIRBKQJIkWagjRBujRFikgRBKT3XkILhPSebHvfmRcTErKbzO6+3ezm3TnHo8e9U95v7sz7Z97MHYUgCAIoEQEiQASIABEgAkTAhgQUJDhsSJeKJgJEgAgQASJABEQCJDjIEYgAESACRIAIEAGbEyDBYXPEVAERIAJEgAgQASJAgoN8gAgQASJABIgAEbA5ARIcNkdMFRABIkAEiAARIAIkOMgHiAARIAJEgAgQAZsTIMFhc8RUAREgAkSACBABIkCCg3yACBABIkAEiAARsDkBEhw2R0wVEAEiQASIABEgAiQ4yAeIABEgAkSACBABmxMgwWFzxFQBESACRIAIEAEiQIKDfIAIEAEiQASIABGwOQESHDZHTBUQASJABIgAESACJDjIB4gAESACRIAIEAGbEyDBYXPEVAERIAJEgAgQASJAgoN8gAgQASJABIgAEbA5ARIcNkdMFRABIkAEiAARIAIkOMgHiAARIAJEgAgQAZsTIMFhc8RUAREgAkSACBABIkCCg3yACBABIkAEiAARsDkBEhw2R0wVEAEiQASIABEgAiQ4yAeIABEgAkSACBABmxMgwWFzxFQBESACRIAIEAEiQIKDfIAIEAEiQASIABGwOQESHDZHTBUQASJABIgAESACJDjIB4gAESACRIAIEAGbEyDBYXPEVAERIAJEgAgQASJAgoN8gAgQASJABIgAEbA5ARIcNkdMFRABIkAEiAARIAIkOMgHiAARIAJEgAgQAZsTIMFhc8RUAREgAkSACBABIkCCg3yACBABIkAEiAARsDkBEhw2R0wVEAEiQASIABEgAiQ4yAeIABEgAkSACBABmxMgwWFzxFQBESACRIAIEAEiQIKDfIAIEAEiQASIABGwOQESHDZHTBUQASJABIgAESACJDjIB4gAESACRIAIEAGbEyDBYXPEVAERIAJEgAgQASJAgoN8gAgQASJABIgAEbA5ARIcNkdMFRABIkAEiAARIAIkOMgHiAARIAJEgAgQAZsTIMFhc8RUAREgAkSACBABIkCCw4gPREVFwWAwWO0doaGhiIyMtLqcolAAscjdi8SDeOQ3rsk/5OsfSqUSISEh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" width="432">





    []




```python

```
