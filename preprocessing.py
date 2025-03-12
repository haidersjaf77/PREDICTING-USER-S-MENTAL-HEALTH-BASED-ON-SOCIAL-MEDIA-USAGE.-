'''
Prepare the dataset by replacing NaN values with 0, selecting features and target for the "depression" label, and checking their shapes.
'''

df_platform = df_platform.replace(np.nan, 0)

label = "depression"
features = [
    "age", "gender", "relationship", "time_spent", 
    "Pinterest", "YouTube", "Reddit", "TikTok", "Instagram", 
    "Facebook", "Snapchat", "Discord", "risk"
]
target = df_platform[label]
data = df_platform[features]

data.shape, target.shape

'''
Encode categorical columns using one-hot encoding and update the dataset by concatenating encoded features with numerical columns.
'''

le = LabelEncoder()
enc_list = ["gender", "relationship", "time_spent", "risk"]
target = le.fit_transform(target)
df_encoded = pd.get_dummies(data[enc_list]).astype(int)
data = pd.concat([data.drop(columns=enc_list), df_encoded], axis=1)
data

