import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, Nadam, Adamax, SGD, RMSprop
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

'''
General settings for warnings, plotting, and DataFrame display.
'''
warnings.filterwarnings('ignore')  
sns.set_style("darkgrid")  
pd.set_option('display.max_columns', None)

df = pd.read_csv("mental_health.csv")
df.head().T

'''
Rename dataset columns for easier accessibility and uniformity
'''

col_names = {
    
    'Timestamp': 'timestamp',
    '1. What is your age?': 'age',
    '2. Gender': 'gender',
    '3. Relationship Status': 'relationship',
    '4. Occupation Status': 'job_status',
    '5. What type of organizations are you affiliated with?': 'affiliation',
    '6. Do you use social media?': 'socialmedia_use',
    '7. What social media platforms do you commonly use?': 'platforms',
    '8. What is the average time you spend on social media every day?': 'time_spent',
    '9. How often do you find yourself using Social media without a specific purpose?': 'purposeless_use',
    '10. How often do you get distracted by Social media when you are busy doing something?': 'distraction',
    "11. Do you feel restless if you haven't used Social media in a while?": 'restless',
    '12. On a scale of 1 to 5, how easily distracted are you?' : 'ease_distract',
    '13. On a scale of 1 to 5, how much are you bothered by worries?' : 'worries',
    '14. Do you find it difficult to concentrate on things?' : 'focus_issue',
    '15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?' : 'comparison',
    '16. Following the previous question, how do you feel about these comparisons, generally speaking?' : 'compare_feel',
    '17. How often do you look to seek validation from features of social media?' : 'validation',
    '18. How often do you feel depressed or down?' : 'depression',
    '19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?' : 'activity_flux',
    '20. On a scale of 1 to 5, how often do you face issues regarding sleep?' : 'sleep_issues'
}

df.rename(columns=col_names, inplace=True)
df.head().T

df.info()

'''
 Handle missing values by replacing NaN in the 'affiliation' column with the most common value
 '''

df['affiliation'].fillna(df['affiliation'].value_counts().index[0], inplace=True)
df.info()

df['gender'].value_counts()

'''
Standardize gender categories by grouping less common values into 'other'
'''
df.gender = df.gender.apply(lambda x: x if x in ["Male","Female"] else "other")
df['gender'].value_counts()