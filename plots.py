'''
Analyze the usage of social media platforms by extracting and counting individual platforms
'''

platform_counts = df.loc[df.socialmedia_use == 'Yes', 'platforms'] \
                    .str.split(', ') \
                    .explode() \
                    .value_counts()

total_usage = platform_counts.sum()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=platform_counts.index, y=platform_counts.values, ax=ax)

ax.set_title("Usage of Different Social Media Platforms", fontsize=16)
ax.set_xlabel("Platforms")
ax.set_ylabel("Count")
ax.tick_params(axis='x', rotation=65)

for p, count in zip(ax.patches, platform_counts.values):
    percent = (count / total_usage) * 100
    ax.annotate(f'n={count}\n{percent:.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height() / 2.), 
                ha='center', va='center')

plt.tight_layout()
plt.show()


'''
One-hot encode social media platforms as individual binary columns
'''

platforms_split = df['platforms'].str.split(', ', expand=True)
platform_list = platforms_split.stack().unique()

for platform in platform_list:
    df[platform] = df['platforms'].str.contains(platform).astype(int)

relevant_features = [
    "age", "gender", "relationship", "job_status", "affiliation", 
    "time_spent", "purposeless_use", "distraction", "restless", 
    "ease_distract", "worries", "focus_issue", "comparison", 
    "compare_feel", "validation", "depression", "activity_flux", 
    "sleep_issues"
] + platform_list.tolist()

df_platform = df[relevant_features]

df_platform.head()

'''
Visualize cumulative social media platform usage vs. user age
'''

fig, ax = plt.subplots(figsize=(10,6))

for platform in platform_list:
    sns.lineplot(x=df_platform.sort_values(by=['age'])["age"], y=df_platform.sort_values(by=['age'])[platform].cumsum(), ax=ax, label=platform)
fig.suptitle("Cumulative platform usage (number of users) vs. age", fontsize=16);

'''
Categorize users into age groups and analyze platform usage within these groups
'''

age_20 = df_platform.query("age <= 20")
age_mid20 = df_platform.query("age > 20 & age <=30")
age_mid30 = df_platform.query("age > 30 & age <=40")
age_40 = df_platform.query("age > 40")

age_groups = {
    "Age Group": ["Age under 20", "Age 21 to 30", "Age 31 to 40", "Age above 40"],
    "Count": [len(age_20), len(age_mid20), len(age_mid30), len(age_40)]
}

age_group_df = pd.DataFrame(age_groups)
age_group_df

'''
showing the relative usage of each social media platform across different age groups.
'''

fig, axes = plt.subplots(3,3, figsize=(10,10))
axes = axes.flatten()

for (ax, platform) in zip(axes, platform_list):
    x_list = ["<=20","21-30","31-40",">=40"]
    percent20 = age_20[platform].sum()/len(age_20)*100
    percent2130 = age_mid20[platform].sum()/len(age_mid20)*100
    percent3140 = age_mid30[platform].sum()/len(age_mid30)*100
    percent40 = age_40[platform].sum()/len(age_40)*100
    y_list = [percent20, percent2130, percent3140, percent40]
    sns.barplot(x=x_list, y=y_list, ax=ax)
    ax.set_ylim(0,100)
    ax.set_title(platform)
fig.suptitle("Relative usage of platforms in age groups", fontsize=16)
plt.tight_layout()

'''
Visualize the distribution of average time spent on social media and its relationship with age and gender.
'''

orderlist = ['Less than an Hour', 'Between 1 and 2 hours', 'Between 2 and 3 hours',
             'Between 3 and 4 hours', 'Between 4 and 5 hours', 'More than 5 hours']

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

sns.barplot(data=df, x=df['time_spent'].unique(), y=df['time_spent'].value_counts(), ax=ax[0], order=orderlist)
ax[0].set_title("Distribution of Average Time Spent per Day on Social Media")
ax[0].set_xlabel("")
ax[0].set_ylabel("Count")
ax[0].tick_params(axis='x', rotation=65)

sns.stripplot(data=df_platform, x=df['time_spent'], y='age', hue='gender', order=orderlist, ax=ax[1])
ax[1].set_title("Insights on Average Time Spent per Day on Social Media vs Age by Gender")
ax[1].set_xlabel("")
ax[1].set_ylabel("Age")
ax[1].tick_params(axis='x', rotation=65)

fig.suptitle("Distribution and Insights on Average Time Spent per Day on Social Media", fontsize=16)

plt.tight_layout()
plt.show()

'''
Calculate and add new columns to the dataset for total platform usage and overall negative impact score.
'''

df_platform["platform_sum"] = 0

for platform in platform_list:
    df_platform["platform_sum"] = df_platform["platform_sum"] + df_platform[platform].fillna(0).astype(int)

df_platform["impact_sum"] = 0

df_platform["impact_sum"] = (
    df_platform["purposeless_use"] + df_platform["distraction"] + df_platform["restless"] + 
    df_platform["ease_distract"] + df_platform["worries"] + df_platform["focus_issue"] + 
    df_platform["comparison"] + df_platform["compare_feel"] + df_platform["validation"] + 
    df_platform["depression"] + df_platform["activity_flux"] + df_platform["sleep_issues"]
)

df_platform.head()

'''
Plot histograms to visualize the distributions of the new columns for platform usage and negative impact.
'''

fig, ax = plt.subplots(1,2, figsize=(10,6))
sns.histplot(data = df_platform, x="platform_sum", ax=ax[0], bins=9)
sns.histplot(data = df_platform, x="impact_sum", ax=ax[1])
fig.suptitle("Distributions of the new columns containing sums", fontsize=14)
ax[0].set_title("Numer of platformes used")
ax[1].set_title("Negative impact (sum)")
plt.tight_layout()

'''
Visualize the relationship between average time spent on social media and its negative impact using different plot types.
'''

fig, ax = plt.subplots(1,2, figsize=(10,6))
sns.stripplot(x=df['time_spent'], y=df_platform['impact_sum'], ax=ax[0], order=orderlist)
sns.boxplot(x=df['time_spent'],  y=df_platform['impact_sum'], ax=ax[0], order=orderlist)
sns.swarmplot(x=df['time_spent'],  y=df_platform['impact_sum'], ax=ax[1], order=orderlist, hue=df_platform['gender'])
ax[0].tick_params(axis='x', rotation=65)
ax[0].set_xlabel("")
ax[1].tick_params(axis='x', rotation=65)
ax[1].set_xlabel("")
fig.suptitle("Negative Impact on User's Mental Health Based on Average Time Spent")
plt.tight_layout()

'''
Display a histogram of the 'impact_sum' column to analyze its distribution.
'''
df_platform['impact_sum'].hist(bins=48)

'''
Visualize 'impact_sum' distribution categorized by risk level ("lower" or "higher") using a histogram.
'''

df_platform["risk"] = "lower"
df_platform.loc[df_platform[df_platform['impact_sum'] >= 37].index, "risk"] = "higher"
df_platform['risk'] = df_platform['risk'].astype("category")
sns.histplot(x=df_platform['impact_sum'], bins=48, hue=df_platform['risk'])
