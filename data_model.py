# %%
import random
import glob 

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# %%
# adjust to compare results
random.seed(0)
#plt.style.use("fivethirtyeight")
drop_duplicates = True

use_data = 1 #  "14_days_after"==1 or "14 days prior" = 14 %%
# configure workspace

files_directory = "/Users/leonahammelrath/FU_Psychoinformatik/Github/blueswatch_open/data"
data_out_directory = "/Users/leonahammelrath/FU_Psychoinformatik/Github/blueswatch_open/outcomes/dataframes"

# import wearable data 

wearable_files = glob.glob("%s/daily_part*.csv" % files_directory)

# import questionnaire data 

questionnaire = pd.read_csv("%s/tk_questionnaire.csv" % data_out_directory)


# %%
questionnaire["date"] = pd.to_datetime(questionnaire.studyInfo_BW, format="%d.%m.%Y")
#questionnaire["date"] = pd.to_datetime(questionnaire.PRE_screening_start, format="%d.%m.%Y")

questionnaire = questionnaire.dropna(subset=["phq_cat", "PRE_gad4"]).copy()
questionnaire["PRE_gad4"] = questionnaire["PRE_gad4"].astype(int)

questionnaire = questionnaire[
    [
        # "TI_MDE",
        # "day",
        # "PRE_height",
        # "PRE_weight",
        "age",
        "PRE_sex",
        "token_BW",
        "date",
        "PRE_phqD1",
        "PRE_phqD2",
        "PRE_gad4",
        "PRE_phqD3",
        "PRE_phqD4",
        "PRE_phqD5",
        "PRE_phqD6",
        "PRE_phqD7",
        "PRE_phqD8",
        "phq_cat",
    ]
]

data = []
for f in wearable_files:
    dataSingle = pd.read_csv(f)
    dataSingle["day"] = pd.to_datetime(dataSingle.day)
    dataSingle.timezoneOffset = (
        dataSingle.timezoneOffset.replace("", "0").astype(float).fillna(0) / 60
    )
   # dataSingle = dataSingle.sort_values(
   #     by=["customer", "source", "type", "day"], #"createdAt"]
   # ).drop_duplicates(keep="last")
    #dataSingle.drop(columns=["createdAt"], inplace=True)
   # dataSingle = dataSingle[
   #     dataSingle.type.isin(["Steps", "HeartRateResting", "SleepDuration"])
   # ]
    data.append(dataSingle)

data = pd.concat(data, ignore_index=True)
#data["customer"] = data.customer.str[:-13]

data = data[data.type.isin(["Steps", "HeartRateResting", "SleepDuration"])]
data = data[["customer", "type", "day", "longValue"]]
steps = data[data.type == "Steps"].copy()
steps = steps.groupby(["customer", "day"], as_index=False).max()
sleep = data[data.type == "SleepDuration"].copy()
sleep = sleep.groupby(["customer", "day"], as_index=False).max()
heart_rate = data[data.type == "HeartRateResting"].copy()
heart_rate = heart_rate.groupby(["customer", "day"], as_index=False).min()

data = pd.concat([steps, sleep, heart_rate])
valid_users = questionnaire.token_BW.unique()
data = data[data.customer.isin(valid_users)].copy()

# %%
results = []
grouped = data.groupby(["customer"])

for name, group in grouped:
    answer_rows = questionnaire[questionnaire.token_BW == name]
    if len(answer_rows) > 1:
        #print(answer_rows)
        if drop_duplicates:
            # only use the last questionnaire date, suggested from Manuel
            answer_rows = answer_rows[
                answer_rows.date == answer_rows.date.max()
            ]
           # print(answer_rows)
    for idx, row in answer_rows.iterrows():
        if use_data == 14:
            group_copy = group[
                (group.day >= row.date - pd.Timedelta(14, unit="day"))
                & (group.day < row.date)
            ].copy()
        elif use_data ==1:
            group_copy = group[
                (group.day >= row.date)
                & (group.day < row.date + pd.Timedelta(14, unit="day"))
            ].copy()
        
        else:
            group_copy = group[
                ((group.day >= row.date - pd.Timedelta(14, unit="day"))
                & (group.day < row.date)) | ((group.day >= row.date)
                & (group.day < row.date + pd.Timedelta(14, unit="day")))
            ].copy()
        

        if len(group_copy.day.unique()) >= 7:
            group_copy["id"] = (
                str(row.token_BW) + "_" + row.date.normalize().strftime("%Y-%m-%d")
            )
            results.append(group_copy)

data = pd.concat(results)

# %%
data.reset_index(drop=True, inplace=True)

# %%
data.to_csv("%s/data_Full_pre_%s.csv" % (data_out_directory, use_data), index=False)

# %%
data_pivot = data.pivot(
    index=["id", "day"],
    columns="type",
    values="longValue",
)

# %%
columns_suff = data_pivot.isnull().sum()[data_pivot.isnull().sum() < 5000].index.values

columns_manual = np.array(
    [
        "Steps",
        "HeartRateResting",
        "SleepDuration",
    ]
)
data_pivot = data_pivot[data_pivot.columns[data_pivot.columns.isin(columns_manual)]]

# %%
data_pivot = data_pivot.reset_index().groupby(["id"])
dataMean = data_pivot.mean().reset_index()
dataStd = data_pivot.std().reset_index()

# %%
dataFull = dataMean.merge(dataStd, how="inner", on=["id"], suffixes=["_mean", "_std"])

# %%
dataFull["customer"] = dataFull.id.str.split("_").str.get(0)
dataFull["day"] = pd.to_datetime(dataFull.id.str.split("_").str.get(1))

# %%
dataFull_final = dataFull.merge(
    questionnaire,
    how="left",
    left_on=["customer", "day"],
    right_on=["token_BW", "date"],
)

data_full_two = dataFull_final.drop(
        columns=[
            "customer",
            "date",
            "day",
            "token_BW",
            # "TI_MDE",
            # "phq_cat",
        ]
    ).reset_index(drop=True)

data_full_two.dropna(inplace=True)
# %%
data_neg = data_full_two[data_full_two.phq_cat == 0]
data_neg_users = list(data_neg.id.unique())
print(len(data_neg_users))

data_pos = data_full_two[data_full_two.phq_cat == 1]
data_pos_users = list(data_pos.id.unique())
print(len(data_pos_users))


dataFull_final.to_csv("%s/data_Full_%s.csv" % (data_out_directory, use_data), index=False)


# %%
if len(data_neg_users) > len(data_pos_users):
    subset_neg = random.sample(data_neg_users, k=len(data_pos.id.unique()))

    data_neg = data_neg[data_neg.id.isin(subset_neg)]
else:
    subset_pos = random.sample(data_pos_users, k=len(data_neg.id.unique()))

    data_pos = data_pos[data_pos.id.isin(subset_pos)]
data_full_two = pd.concat([data_neg, data_pos]).reset_index(drop=True)

print("number of total users: %d" % len(data_full_two.index))
print(data_full_two.columns.tolist())
# %%
predict_columns = data_full_two.columns.values[1:]
result_columns = data_full_two.columns.values[[-1]]
predict_columns = np.setdiff1d(predict_columns, result_columns)

# %%
predict_columns = np.array(predict_columns, dtype=str)
categorical_features = predict_columns[np.char.startswith(predict_columns, "PRE")]

numerical_features = np.setdiff1d(predict_columns, categorical_features)

# %%
numeric_transformer = Pipeline(
    steps=[
        ("scaler_final", preprocessing.StandardScaler()),
        ("min_max_scaler_final", preprocessing.MaxAbsScaler()),
    ]
)
column_trans = ColumnTransformer(
    [
        (
            "numerical_features",
            numeric_transformer,
            numerical_features,
        ),
        (
            "categorical_features",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ],
    remainder="passthrough",
    #verbose_feature_names_out=False,
    sparse_threshold=0,
)
X_train = data_full_two.loc[:, predict_columns]

X_train = column_trans.fit_transform(X_train)


X_train = pd.DataFrame(X_train, columns=column_trans.get_feature_names_out())

y_train = data_full_two.loc[:, result_columns]

clf = lgb.LGBMClassifier(
    boosting_type="dart",
    random_state=0,
    # n_estimators=2500,
    # learning_rate=0.05,
)
clf.fit(
    X_train,
    np.ravel(y_train),
)

# clf.save_model("model.txt", num_iteration=clf.best_iteration)

joblib.dump(clf, "model_new.pkl")
joblib.dump(column_trans, "transformer_new.pkl")
# %%
# %%
pred = clf.predict(X_train)
data_full_two["prediction"] = pred
# %%
print(data_full_two[data_full_two.prediction == 1].dropna().head(5).to_string())
# %%
idx = data_full_two[data_full_two.prediction == 1].dropna().head(5).index
print(X_train.loc[idx].to_string())