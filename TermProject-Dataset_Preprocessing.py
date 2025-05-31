#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv(r'C:\AB_NYC_2019.csv')
df


# In[3]:


df.isnull().sum()


# In[4]:


df.dtypes


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.columns


# In[9]:


null_rows = df[df['name'].isnull()]
(null_rows)


# In[10]:


null_host_name = df[df['host_name'].isnull()]
(null_host_name)


# host_name이랑 name 둘다 null 인거 없음 --> 그냥 drop ㄱㄱ

# In[11]:


null_host_name_and_name = df[(df['host_name'].isnull()) & (df['name'].isnull())]
(null_host_name_and_name)


# 여기서 drop하는거 작성

# In[12]:


df2 = df.copy()


# 여기는 last_review 컬럼이 문자열 데이터 이기 때문에 timestamp형태로 바꿔줌

# In[13]:


df2['last_review'] = pd.to_datetime(df2['last_review'], errors='coerce')


# In[14]:


# 오늘 날짜에서 얼마나 지났는지 계산해서 days_since_review column만듬 --> 마지막 리뷰 대체
# na값은 --> 9999로 설정
today = pd.to_datetime("today")

df2['days_since_review'] = (today - df2['last_review']).dt.days
df2['days_since_review'] = df2['days_since_review'].fillna(9999)



# In[15]:


df2.isnull().sum()


# In[16]:


# last_review column drop하기
df2 = df2.drop(columns =['last_review'])


# In[17]:


# reviews_per_month의 na값들은 0으로 바꿔자
df2['reviews_per_month'] = df2['reviews_per_month'].fillna(0)


# In[18]:


df2.isnull().sum()


# In[19]:


df2


# In[20]:


# host_id=na 이면서 days_since_review = 9999 이면 drop하자 허위매물 같음, 마찬가지로 name = na값이면서 리뷰 없는거도 drop
# 그리고 중간에 보면 minimum nights가 1000인것도 있는데 feature별로 outlier있는지 파악 해봐야할듯


# In[21]:


index_to_drop = df2[(df2['host_name'].isnull()) &(df2['days_since_review'] == 9999)].index


# In[22]:


index_to_drop


# In[23]:


df2.drop(index = index_to_drop, inplace=True)


# In[24]:


df2[df2['host_name'].isnull()]


# In[25]:


df2[df2['name'].isnull()]


# In[26]:


index_to_drop_name = df2[(df2['name'].isnull()) &(df2['days_since_review'] == 9999)].index


# In[27]:


df2.drop(index = index_to_drop_name, inplace=True)


# In[28]:


df2[df2['name'].isnull()|df2['host_name'].isnull()]


# name이랑 hostname null값이지만 유의미하다고 생각하는 데이터들 unknown으로 우선 변경

# In[29]:


df2['name'] = df2['name'].fillna('unknown')


# In[30]:


df2['host_name'] = df2['host_name'].fillna('unknown')


# In[31]:


df2.isnull().sum()


# In[32]:


df2.describe()


# In[33]:


df2


# In[34]:


df2['neighbourhood_group'].unique()


# In[35]:


df2['room_type'].unique()


# In[36]:


df2['neighbourhood'].value_counts(dropna = False)


# In[37]:


df2['neighbourhood'].value_counts().head(10)


# In[38]:


#air bnb 데이터라 price가 0인건 의미가 없다고 판단
df2[df2['price']<=0]


# In[39]:


df2=df2[df2['price']>0]


# In[40]:


# minimum nights-->이거 IQR로 보자


# In[41]:


df2[df2['availability_365']>365]


# In[42]:


df2[df2['number_of_reviews']<0]


# In[43]:


df2[df2['minimum_nights']>365]


# minimum night이 터무니 없이 크지만 리뷰수는 없다? 제거 필요하다고 생각

# In[44]:


Q1 = df2['minimum_nights'].quantile(0.25)
Q3 = df2['minimum_nights'].quantile(0.75)
IQR = Q3 - Q1

df2[df2['minimum_nights'] <= Q1 - 1.5 * IQR]


# In[45]:


df2[df2['minimum_nights'] > Q3 + 1.5 * IQR]


# IQR이랑 비교하긴 좀 그렇지만 암튼 minimum night이랑 리뷰수 0인거 drop하자 조건 잘 맞춰서 

# In[46]:


len(df2[df2['minimum_nights'] > Q3 + 1.5 * IQR])
# 생각보다 상위 이상치가 너무 많이 존재한


# In[47]:


import matplotlib.pyplot as plt


# In[48]:


outlier_mask = df2['minimum_nights'] > Q3 + 1.5 * IQR

outlier_min_nights = df2.loc[outlier_mask, 'minimum_nights']


# In[49]:


counts, bins, _ = plt.hist(outlier_min_nights, bins=20, edgecolor='black')
plt.title('outlier for min_nights')
plt.xlabel('min_nights')
plt.ylabel('frequency')
plt.show()

# 첫 번째 bin의 경계 출력

print(f"첫 번째 bin 범위: {bins[0]:.2f} ~ {bins[1]:.2f}")


# 73.9이상은 drop 하자

# In[50]:


df2[df2['minimum_nights']>73.9]


# In[51]:


df2 = df2[df2['minimum_nights']<=73.9]


# In[52]:


df2


# price랑 리뷰수도 같이봐야할듯?

# In[53]:


price_detection = df2['price'].to_numpy()
plt.hist(price_detection,edgecolor = 'black')
plt.title('price distribution')
plt.xlabel('price')
plt.ylabel('frequency')
plt.show()


# In[54]:


Q1_price = df2['price'].quantile(0.25)
Q3_price = df2['price'].quantile(0.75)
IQR_price = Q3_price - Q1_price

df2[df2['price'] <= Q1_price - 1.5 * IQR_price]


# In[55]:


outlier_price = df2[df2['price']>=Q3_price +1.5 * IQR_price]
outlier_price


# In[56]:


counts, bin_edges, _ = plt.hist(outlier_price['price'], bins=20, edgecolor='black')
plt.title('Outlier Price Distribution')
plt.xlabel('price')
plt.ylabel('frequency')
plt.show()

print(f"첫 번째 bin 범위: {bin_edges[0]:.2f} ~ {bin_edges[1]:.2f}")


# In[57]:


outlier_price[(outlier_price['price']>817.3) & (outlier_price['number_of_reviews']==0)]


# In[58]:


outlier_price[outlier_price['neighbourhood_group']=='Manhattan']


# #### 여기까지 한 결과 단순히 price로만 고려하면 안됌 지역별로 묶어서 price고려하기로 결정

# In[59]:


k =df2.groupby('neighbourhood_group')
k['price'].mean()


# In[60]:


def iqr_outlier_mask(group):
    Q1 = group['price'].quantile(0.25)
    Q3 = group['price'].quantile(0.75)
    IQR = Q3 - Q1
    return (group['price'] > Q3 + 1.5 * IQR)

# 각 그룹별로 적용
df2['is_outlier_price'] = df2.groupby('neighbourhood_group').apply(iqr_outlier_mask).reset_index(level=0, drop=True)


# In[61]:


a = df2[df2['is_outlier_price']==True]


# In[62]:


# 도시별 outlier들 시각화 어디까지 자를것인가
group_a = a.groupby('neighbourhood_group')
for name, group in group_a:
    plt.hist(group['price'], bins = 50, edgecolor = 'black')
    plt.title(f"price distribution for {name}")
    plt.xlabel('price')
    plt.ylabel('frequency')
    plt.show()



# In[63]:


def get_extreme_outliers(group):
    Q1 = group['price'].quantile(0.25)
    Q3 = group['price'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    return group[group['price'] > upper_bound]
extreme_outlier = group_a.apply(get_extreme_outliers).reset_index(drop=True)
extreme_outlier


# In[64]:


#outlier 의 outlier
group_extreme_outlier = extreme_outlier.groupby('neighbourhood_group')
for name, group in group_extreme_outlier:
    plt.hist(group['price'], bins = 50, edgecolor = 'black')
    plt.title(f"price distribution for {name}")
    plt.xlabel('price')
    plt.ylabel('frequency')
    plt.show()



# In[65]:


k['price'].mean()


# In[66]:


#다시 index df2랑 맞춰야함 인덱스 정상화 작업
extreme_outlier = group_a.apply(get_extreme_outliers)
extreme_outlier


# In[67]:


#drop 해줌
drop_index = extreme_outlier.index.get_level_values(-1)
df2 = df2.drop(index=drop_index)
df2


# In[68]:





# encoding 시작

# In[69]:


df2 = pd.get_dummies(df2, columns =['neighbourhood_group'], drop_first=True, dtype = int)
df2


# In[70]:


df2 = pd.get_dummies(df2, columns =['room_type'], drop_first=True, dtype = int)
df2


# In[78]:


# 실제 남은 숙소 수 기준으로 다시 계산
# drop시킨 데이터들의 개수 감소가 calculated_host_listings_count의 감소로 이어지지 않아 예측에 영향을 미칠경우 이거 사용해서 또 돌려보자!
#actual_counts = df2['host_id'].value_counts()
#df2['adjusted_listings_count'] = df2['host_id'].map(actual_counts)


# 학습에 필요한 feature들 Standard scailing 진행

# In[82]:


from sklearn.preprocessing import StandardScaler
df3 = df2.copy()
scale_features = ['minimum_nights', 'availability_365', 'calculated_host_listings_count']
scaler = StandardScaler()
df3[scale_features] = scaler.fit_transform(df3[scale_features])
df3


# In[89]:


# price분포가 너무 크기 떄문에 regression 예측시 편향 줄여줌
df2['log_price'] = np.log1p(df2['price'])
df2


# In[88]:


#나중에 예측값 price로 돌릴때 np.expm1(y_pred)로 돌려서 최종 값이랑 비교


