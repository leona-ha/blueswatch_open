# %%

import glob
import time
import numpy as np

import pandas as pd

# configure workspace

files_directory = "/Users/leonahammelrath/FU_Psychoinformatik/Github/blueswatch_open/data"
data_out_directory = "/Users/leonahammelrath/FU_Psychoinformatik/Github/blueswatch_open/outcomes/dataframes"


# import wearable data 

wearable_files = glob.glob("%s/daily_part*.csv" % files_directory)

use_data = 1 #  "all"==1 or "14_days_prior"=14 or bw3= 3
drop_duplicates = True 

data = []
for f in wearable_files:
    dataSingle = pd.read_csv(f)
    dataSingle["day"] = pd.to_datetime(dataSingle.day)
    dataSingle.timezoneOffset = (
        dataSingle.timezoneOffset.replace("", "0").astype(float).fillna(0) / 60
    )
    dataSingle = dataSingle.sort_values(
        by=["customer", "source", "type", "day"], #"createdAt"]
    ).drop_duplicates(keep="last")
    dataSingle.drop(columns=["createdAt"], inplace=True)
    dataSingle = dataSingle[
        dataSingle.type.isin(["Steps", "HeartRateResting", "SleepDuration"])
    ]
    data.append(dataSingle)

data = pd.concat(data, ignore_index=True)
data["customer"] = data.customer.str[:-13]



print("unique customers with data: %d" % len(data.customer.unique()))


# %%
answers = pd.read_csv("%s/answers.csv" % files_directory)
answers["user"] = answers.user.str[:-13]
questionnaire_sessions = pd.read_csv(
    "%s/questionnaireSession.csv" % files_directory
).dropna(subset=["completedAt"])
questionnaire_sessions.completedAt = pd.to_datetime(
    questionnaire_sessions.completedAt, unit="ms"
)
questionnaire_sessions["day"] = questionnaire_sessions.completedAt.dt.normalize()
questionnaire_sessions["user"] = questionnaire_sessions.user.str[:-13]
choice = pd.read_csv("%s/choice.csv" % files_directory)

print(
    "unique customers with full questionnaires BW3: %d"
    % len(
        questionnaire_sessions[questionnaire_sessions.questionnaire == 33].user.unique()
    )
)

print(
    "unique customers with full questionnaires BW4: %d"
    % len(
        questionnaire_sessions[questionnaire_sessions.questionnaire == 45].user.unique()
    )
)

# %%
valid_users = questionnaire_sessions[
    (questionnaire_sessions.questionnaire == 33)|(questionnaire_sessions.questionnaire == 45)
].user.unique()
data = data[data.customer.isin(valid_users)]

print(
    "unique customers with any kind of daily data who also completed survey: %d"
    % len(data.customer.unique())
)
# %%

# %%
results = []
grouped = data.groupby(["customer", "source"])
for name, group in grouped:
    answer_rows = questionnaire_sessions[
        ((questionnaire_sessions.user == name[0])
        & (questionnaire_sessions.questionnaire == 33)) | ((questionnaire_sessions.user == name[0])
        & (questionnaire_sessions.questionnaire == 45))
    ]
    if len(answer_rows) > 1:
        #print(answer_rows)
        if drop_duplicates:
            # only use the last questionnaire date, suggested from Manuel
            answer_rows = answer_rows[
                answer_rows.completedAt == answer_rows.completedAt.max()
            ]
            #print(answer_rows)
    for idx, row in answer_rows.iterrows():
        if use_data == 14: # check if enough data are available when only including data before quest
            group_copy = group[
                (group.day >= row.day - pd.Timedelta(14, unit="day"))
                & (group.day < row.day)
            ].copy()
        else:
            group_copy = group[
                (group.day >= row.day)
                & (group.day < row.day + pd.Timedelta(14, unit="day"))
            ].copy()
        if len(group_copy.day.unique()) >= 7: # check if 50% of data are available
            group_copy["id"] = (
                str(row.user)
                + "_"
                + name[1]
                + "_"
                + row.day.normalize().strftime("%Y-%m-%d")
            )
            results.append(group_copy)

data = pd.concat(results)
print(len(data.customer.unique()))
# %%
data.reset_index(drop=False, inplace=True)

# %%

# %%
data = data[~data.type.isin(["SleepStartTime", "SleepEndTime"])].copy()

idx = data[data.valueType == 10].index
data.loc[idx, "longValue"] = data.loc[idx, "doubleValue"]

data.to_csv("%s/data_pre_%s.csv" % (data_out_directory, use_data), index=False)

data.drop(columns=["valueType", "doubleValue", "dateValue", "booleanValue", "stringValue", \
"generation", "trustworthiness", "medicalGrade", "chronologicalExactness", "userReliability"], inplace=True)

print(data.groupby(["id", "type"])["day"].count().mean())


data_pivot = data.pivot_table(
    index=["id", "day"],
    columns="type",
    values="longValue" #, aggfunc=np.mean
)

columns_manual = np.array(
    [
        "Steps",
        "HeartRateResting",
        "SleepDuration",
    ]
)
data_pivot = data_pivot[data_pivot.columns[data_pivot.columns.isin(columns_manual)]]

data_pivot = data_pivot.reset_index().groupby(["id"])

dataMean = data_pivot.mean().reset_index()
dataStd = data_pivot.std().reset_index()

dataFull = dataMean.merge(dataStd, how="inner", on=["id"], suffixes=["_mean", "_std"])
dataFull["customer"] = dataFull.id.str[:3]
dataFull["day"] = pd.to_datetime(dataFull.id.str.split("_").str.get(2))

dataFull.to_csv("%s/data_Full_pre_%s.csv" % (data_out_directory, use_data), index=False)
