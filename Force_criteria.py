#%%
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#讀取資料
data = pd.read_excel(r"C:\Users\user\OneDrive - Lehigh University\Desktop\investment\python\API\TX00\TX00_5_20241101_20241212.xlsx")
data = data.loc[~(data == 0).any(axis=1)]
data.index.name = 'Datetime'



#%%
from back_testing.Backtest import RealtimeSpeedForceIndicators

ForceIndicator = RealtimeSpeedForceIndicators(speed_window=3,force_window=3,
                                              range_window=3,max_allowed_std=50000)

ForceIndicator.add_new_data(data)
data_test = ForceIndicator.data

# Classify time periods
def classify_time_period(row):
    time = row.name.time()
    if time >= pd.to_datetime("08:45").time() and time <= pd.to_datetime("13:45").time():
        return "day_session"
    elif time >= pd.to_datetime("15:00").time() or time <= pd.to_datetime("05:00").time():
        return "night_session"
    else:
        return None
# Convert datetime to pandas datetime format
data_test["datetime"] = pd.to_datetime(data["date"])
data_test.set_index("datetime", inplace=True)
data_test["session"] = data_test.apply(classify_time_period, axis=1)
data_test = data_test.dropna(subset=["session"])
N = 3  # 例如，預測未來 3 期的回報率
data_test["future_return"] = data_test["close"].shift(-N) / data_test["close"] - 1
data_test = data_test.dropna(subset=["future_return"])

# Define thresholds and label mapping
up_threshold = 0.0004
down_threshold = -0.0008
weak_up_threshold = 0.0002
weak_down_threshold = -0.0004
label_mapping = {-1: 0, -0.5: 1, 0: 2, 0.5: 3, 1: 4}

# Create trend labels
data_test["trend_label"] = 0.0
data_test.loc[data_test["future_return"] > up_threshold, "trend_label"] = 1  # Strong up
data_test.loc[data_test["future_return"] < down_threshold, "trend_label"] = -1  # Strong down
data_test.loc[(data_test["future_return"] > weak_up_threshold) & (data_test["future_return"] <= up_threshold), "trend_label"] = 0.5  # Weak up
data_test.loc[(data_test["future_return"] < weak_down_threshold) & (data_test["future_return"] >= down_threshold), "trend_label"] = -0.5  # Weak down
data_test["trend_label"] = data_test["trend_label"].map(label_mapping).astype(int)

# Select features
features = ["open", "high", "low", "close", "volume", "speed", "force", "smoothed_range_diff", "force_std", "range_diff"]
for col in features:
    data_test[col] = pd.to_numeric(data_test[col], errors="coerce")

data_test = data_test.dropna(subset=features)

# Separate sessions
day_session = data_test[data_test["session"] == "day_session"]
night_session = data_test[data_test["session"] == "night_session"]

# Prepare training and testing data_test
X_day = day_session[features]
y_day = day_session["trend_label"]
X_night = night_session[features]
y_night = night_session["trend_label"]

X_train_day, X_test_day, y_train_day, y_test_day = train_test_split(X_day, y_day, test_size=0.3, random_state=42, stratify=y_day)
X_train_night, X_test_night, y_train_night, y_test_night = train_test_split(X_night, y_night, test_size=0.3, random_state=42, stratify=y_night)

# %% Build and train models
from collections import Counter
# Step 5: 訓練模型（使用類別加權）
class_weights = Counter(y_train_day)
total = sum(class_weights.values())
class_weights = {k: total / v for k, v in class_weights.items()}

# 自定義損失函數的加權設置
xgb_model_day = XGBClassifier(
    random_state=42,
    eval_metric="mlogloss",
    n_estimators=300,
    learning_rate=0.01,
    max_depth=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    use_label_encoder=False,
    scale_pos_weight=class_weights[4]  # 設置加權
)

xgb_model_day.fit(X_train_day, y_train_day)

# Night session 模型
xgb_model_night = XGBClassifier(
    random_state=42,
    eval_metric="mlogloss",
    n_estimators=300,
    learning_rate=0.01,
    max_depth=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    use_label_encoder=False
)

xgb_model_night.fit(X_train_night, y_train_night)

# Step 6: 評估模型
reverse_mapping = {0: "Down (-1)", 1: "Weak Down (-0.5)", 2: "Neutral (0)", 3: "Weak Up (0.5)", 4: "Up (1)"}

# Day session
y_pred_day = xgb_model_day.predict(X_test_day)
print("Day Session Classification Report:")
print(classification_report(y_test_day, y_pred_day, target_names=[reverse_mapping[label] for label in sorted(reverse_mapping.keys())]))
print("Day Session Confusion Matrix:")
print(confusion_matrix(y_test_day, y_pred_day))

# Night session
y_pred_night = xgb_model_night.predict(X_test_night)
print("\nNight Session Classification Report:")
print(classification_report(y_test_night, y_pred_night, target_names=[reverse_mapping[label] for label in sorted(reverse_mapping.keys())]))
print("Night Session Confusion Matrix:")
print(confusion_matrix(y_test_night, y_pred_night))

# Step 7: 特徵重要性
feature_importances_day = pd.DataFrame({
    "Feature": features,
    "Importance": xgb_model_day.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importances_night = pd.DataFrame({
    "Feature": features,
    "Importance": xgb_model_night.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nDay Session Feature Importances:")
print(feature_importances_day)

print("\nNight Session Feature Importances:")
print(feature_importances_night)

# 可視化
plt.figure(figsize=(12, 6))
plt.barh(feature_importances_day["Feature"], feature_importances_day["Importance"], label="Day Session")
plt.barh(feature_importances_night["Feature"], feature_importances_night["Importance"], label="Night Session", alpha=0.7)
plt.title("Feature Importances: Day vs Night Sessions")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.legend()
plt.tight_layout()
plt.show()
# %%

# 提取特徵和分類數據範圍
selected_features = ["speed", "force", "smoothed_range_diff", "force_std", "range_diff"]
classification_ranges_day = {}
classification_ranges_night = {}

# 計算日盤每個分類的特徵中位數
for label in sorted(reverse_mapping.keys()):  # reverse_mapping 包含標籤名稱
    subset_day = day_session[day_session["trend_label"] == label][selected_features]
    classification_ranges_day[reverse_mapping[label]] = subset_day.median()

# 計算夜盤每個分類的特徵中位數
for label in sorted(reverse_mapping.keys()):  # reverse_mapping 包含標籤名稱
    subset_night = night_session[night_session["trend_label"] == label][selected_features]
    classification_ranges_night[reverse_mapping[label]] = subset_night.median()

# 將結果轉為DataFrame便於查看
import pandas as pd

median_day_df = pd.DataFrame(classification_ranges_day).transpose()
median_night_df = pd.DataFrame(classification_ranges_night).transpose()
# %%
# 提取特徵和分類數據範圍
selected_features = ["speed", "force", "smoothed_range_diff", "force_std", "range_diff"]
classification_ranges = {}

# 計算每個分類的特徵統計數據
for label in sorted(reverse_mapping.keys()):  # reverse_mapping 包含標籤名稱
    subset = data_test[data_test["trend_label"] == label][selected_features]
    classification_ranges[reverse_mapping[label]] = subset.describe()

# 顯示每個分類的統計數據
for label, stats in classification_ranges.items():
    print(f"Classification: {label}")
    print(stats)

# %%
