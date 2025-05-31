import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# 2025.05.24. (202135844 최우진)
# Data Science Term Project
# Modeling - Analysis - Evaluation Process

# Model Selection : Linear Model (다중 선형 회귀 모델)
# Analysis Process :
# First - Date set split -> Train : Test = 8 : 2
# Input Features - Total 7 : categorical 6 + number_of_reviews
# 1) 4 neighbourhood_groups - categorical value - onehot encoded (Brooklyn, Manhattan, Queens, State Island) 
# 2) 2 room_types - categorical value - onehot encoded (private room, shared room)
# 3) number_of_reviews - integer value - minmax scaling으로 scaling
# min = 0, max = 629
# Target Variable - log_price : price in log


# Visualization Process : matplotlib
# -> price value between : predicted log_price vs actual log_price   
# Predict function : predict_price
# -> input : array of onehot encoded features + 'number_of_reviews'
# -> then print - predicted price in $

# Evaluation Process :
# 1. Evaluation Methods : K-Fold validation
# -> K = 5
# 2. Evaluation Metrics : MSE , RMSE

# csv 파일 read
df = pd.read_csv(r'C:\clean_NYC_v1.csv')

# minmax scaling을 이용해서 number_of_reviews의 값 스케일링
mms_reviews = MinMaxScaler()
df[['number_of_reviews']] = mms_reviews.fit_transform(df[['number_of_reviews']])

# feature = X, target = Y
# X1 = onehot encoding된  
# 4개의 neighbourhood columns + 2개의 room type columns
X1 = df.iloc[:, 16:-1] 
X2 = df[['number_of_reviews']]
X = pd.concat([X1, X2], axis=1)  
Y = df[['log_price']]  

# split dataset into train & test dataset (8 : 2)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


reg = LinearRegression()
reg.fit(X_train, y_train)

# 예측 및 예측값 저장

y_pred = reg.predict(X_test)

#Evaluation metrics -> mse, rmse 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_score = cross_val_score(reg, X, Y, cv = 10)

print("Evaluation Metrics:")
print("Mean Squared Error (MSE):", mse, '\n')


print('cross_validation 점수 : ', cv_score, '\n')
print('평균 cross_validation 점수 : ', cv_score.mean(), '\n')

# visualization - predicted price vs actual price
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue', label='Predicted & Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='Perfect Prediction')
plt.xlabel("Actual log-price")
plt.ylabel("Predicted log-price")
plt.title("Predicted & Actual Log-Prices")
plt.legend()
plt.grid(True)
plt.show()

# 데이터를 넣으면 price값 predict 후 np.expm1을 통해 다시 $값으로 복원하여 출력

def predict_price(onehot_array, num_review):
    # onehot_array: 인코딩된 6개 컬럼 (neighborhood & room type)
    # num_review: 원본 리뷰 수
    review_scaled = mms_reviews.transform([[num_review]])[0, 0]
    input_data = np.array(onehot_array + [review_scaled]).reshape(1, -1)
    log_prediction = reg.predict(input_data)
    price = np.expm1(log_prediction[0, 0])
    print(f"예상 가격: {price:.2f} USD")

# example -> onehot_array = OneHot 인코딩된 값들 - [neighbourhood_group_Brooklyn, Manhatten, Queens, State Island, room_type_Private room, room_type_Shared room]
# predict_price([0,0,1,0,0,1], 25)  

# K-Fold : K = 5
kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = []
mse_scores = []

# K-Fold validation
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = Y.iloc[train_idx], Y.iloc[test_idx]

    reg_fold = LinearRegression()
    reg_fold.fit(X_train_fold, y_train_fold)

    y_pred_fold = reg_fold.predict(X_test_fold)

    r2 = r2_score(y_test_fold, y_pred_fold)
    mse = mean_squared_error(y_test_fold, y_pred_fold)

    r2_scores.append(r2)
    mse_scores.append(mse)

    print(f"Fold {fold+1}")
    print(f"R²: {r2:.4f}, MSE: {mse:.6f}\n")

# Average peformance of 5-fold
print("K-Fold 평균 성능")
print(f"평균 R²: {np.mean(r2_scores):.4f}")
print(f"평균 MSE: {np.mean(mse_scores):.6f}")
