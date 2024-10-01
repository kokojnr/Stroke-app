import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

penguin_df = pd.read_csv("healthcare-dataset-stroke-data.csv")
output = penguin_df["stroke"]
penguin_df.bmi.fillna(penguin_df.bmi.mean(),inplace=True)
features = penguin_df[
    [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "avg_glucose_level",
        "work_type",
        "stroke"
    ]
]
df = features.copy()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in df.columns:
    if df[col].dtype=='object':
        df[col]=le.fit_transform(df[col])

from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd
X = df.drop("stroke",axis=1)
y =df['stroke']


# Check the original class distribution
print('Original dataset shape:', Counter(y))

# Initialize RandomUnderSampler
rus = RandomUnderSampler(random_state=42)

# Perform undersampling
X_resampled, y_resampled = rus.fit_resample(X, y)

# Check the resampled class distribution
print('Resampled dataset shape:', Counter(y_resampled))
features = df[features.columns]
output, uniques = pd.factorize(output)

x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3)
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train.values, y_train)
y_pred = rfc.predict(x_test.values)
score = accuracy_score(y_pred, y_test)
print("Our accuracy score for this model is {}".format(score))
print(X.shape)


rf_pickle = open("random_forest_penguin.pickle", "wb")
pickle.dump(rfc, rf_pickle)
rf_pickle.close()
output_pickle = open("output_penguin.pickle", "wb")
pickle.dump(uniques, output_pickle)
output_pickle.close()

