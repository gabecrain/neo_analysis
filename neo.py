import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import trim_mean, iqr, pearsonr, stats
import statsmodels.api as sm

neo = pd.read_csv('/Users/gabecrain/Desktop/NEO_Project/NEO.csv')
# print(neo.head())

print(neo.columns.tolist())
neo = neo.drop(['orbiting_body', 'name', 'sentry_object'], axis=1)
#drop columns that arent relevant to the analysis

#univariate analysis of miss_distance, relative_velocity, est_diameter_min, hazardous

#miss_distance
mean_miss_distance = neo.miss_distance.mean()
trim_mean_miss = trim_mean(neo.miss_distance, proportiontocut=.05)
median_miss_distance = neo.miss_distance.median()

print('mean miss distance:', mean_miss_distance)
print('trim mean miss distance:', trim_mean_miss)
print('median miss distance:', median_miss_distance)
print('\n')

#relative_velocity
mean_relative_velocity = neo.relative_velocity.mean()
trim_mean_relative_velocity = trim_mean(neo.relative_velocity, proportiontocut=.05)
median_relative_velocity = neo.relative_velocity.median()

print('mean relative velocity:', mean_relative_velocity)
print('trim mean relative velocity:', trim_mean_relative_velocity)
print('median relative velocity:', median_miss_distance)
print('\n')

#est_diameter_min
mean_est_diameter_min = neo.est_diameter_min.mean()
trim_mean_est_diameter_min = trim_mean(neo.est_diameter_min, proportiontocut=.05)
median_est_diameter_min = neo.est_diameter_min.median()

print('mean minimum estimated diameter:', mean_est_diameter_min)
print('trim mean minimum estimated diameter:', trim_mean_est_diameter_min)
print('median minimum estimated diameter:', median_est_diameter_min)
print('\n')

#hazardous
hazardous_counts = neo.hazardous.value_counts()
hazardous_props = neo.hazardous.value_counts(normalize=True)
print('the value counts for the hazardous column:\n', hazardous_counts)
print('the value proportions for the hazardous column:\n', np.round(hazardous_props*100, 2))

print('\n')

#visual analysis of miss_distance, relative_velocity, est_diameter_min, hazardous

#miss_distance
sns.boxplot(x='miss_distance', data=neo)
plt.xlabel('Object Miss Distance (km)')
plt.title('Object Miss Distance From Earth')
# plt.show()
plt.clf()

sns.histplot(x='miss_distance', data=neo)
plt.xlabel('Object Miss Distance (km)')
plt.ylabel('Object Frequency')
plt.title('Relationship between Object Count and Miss Distance')
plt.axvline(mean_miss_distance, color='r', linestyle='solid', linewidth=2, label='Mean')
plt.axvline(median_miss_distance, color='y', linestyle='solid', linewidth=2, label='Median')
plt.legend()
# plt.show()
plt.clf()
#the majority of the objects had a miss distance just under 40000000 kms. There was also a high concentration of objects recorded with a much shorter miss distance, bringing down the overall average.
#it is possible this is because it is easier to observe and spot objects that are in closer proximity to the earth. Furthermore it would be prudent to study near objects that have a higher probibility of striking the earth.


#relative_velocity
sns.boxplot(x='relative_velocity', data=neo)
plt.xlabel('Frequency of Relative Velocity')
plt.title('Frequency of Object Relative Velocity')
# plt.show()
plt.clf()

#find quantiles of relative velocity data
relative_velocity_quartiles = np.quantile(neo.relative_velocity, [.25, .5, .75])
print('relative velocity quantiles:', np.round(relative_velocity_quartiles))

sns.histplot(x='relative_velocity', data=neo)
plt.xlabel('Object Relative Velocity (kms/hr)')
plt.ylabel('Frequency')
plt.title('Frequency of Object Relative Velocity')
plt.axvline(relative_velocity_quartiles[0], color='r', linestyle='solid', linewidth=2, label='Q1')
plt.axvline(relative_velocity_quartiles[1], color='orange', linestyle='solid', linewidth=2, label='Q2')
plt.axvline(relative_velocity_quartiles[2], color='y', linestyle='solid', linewidth=2, label='Q3')
plt.legend()
# plt.show()
plt.clf()
#the majority of the objects had a velocity near 35000 kms/s with some extreme outliers faster than 150000 kms/s

#separate relative_velocity data by median, take quartiles of each subset and plot a histogram of the data
low_relative_velocity = neo.relative_velocity[neo.relative_velocity <= median_relative_velocity]
high_relative_velocity = neo.relative_velocity[neo.relative_velocity >= median_relative_velocity]

low_velocity_quartiles = np.quantile(low_relative_velocity, [.25, .5, .75])
print('low relative velocity quantiles:', low_velocity_quartiles)

high_velocity_quartiles = np.quantile(high_relative_velocity, [.25, .5, .75])
print('high relative velocity quantiles:', high_velocity_quartiles)

plt.hist(low_relative_velocity, alpha=.6, bins=50, label='Low Velocity')
plt.hist(high_relative_velocity, alpha=.6, bins=50, label='High Velocity')
plt.legend()
plt.show()
plt.clf()

print('\n')

# #est_diameter_min
neo['trimmed_diameter_min'] = neo.est_diameter_min < .1
sns.boxplot(x='trimmed_diameter_min', data=neo)
# plt.show()
plt.clf()
# #visualization was hard to interpret due to extreme outliers. I removed the extreme 5% outliers from either side.

sns.histplot(x='trimmed_diameter_min', data=neo)
# plt.show()
plt.clf()


#hazardous
sns.countplot(x='hazardous', data=neo)
plt.xlabel('Hazardous')
plt.ylabel('Object Count')
plt.title('Hazardous vs. Non-Hazardous')
# plt.show()
plt.clf()

# neo.hazardous.value_counts().plot.pie(autopct='%.1f')
plt.title('Number of Objects Considered Hazardous')
# plt.show()
plt.clf()
#the majority of neo objects in this dataset are not considered hazardous, with less than 10% of entries labeled as hazardous.


#analyzing relationship between hazardous and velocity variables
true_hazardous = neo.relative_velocity[neo.hazardous == True]
false_hazardous = neo.relative_velocity[neo.hazardous == False]

mean_hazardous_velocity = np.mean(true_hazardous)
mean_non_hazarous_velocity = np.mean(false_hazardous)
median_hazardous_velocity = np.median(true_hazardous)
median_non_hazardous_velocity = np.median(false_hazardous)

print('mean velocity of objects classified as hazardous:', mean_hazardous_velocity)
print('mean velocity of objects classified as non hazardous:', mean_non_hazarous_velocity)
print('difference between mean velocity of objects classified as hazardous vs non hazardous:', mean_hazardous_velocity - mean_non_hazarous_velocity)

print('\n')

print('median velocity of objects classified as hazardous:', median_hazardous_velocity)
print('median velocity of objects classified as non hazardous:', median_non_hazardous_velocity)
print('difference between median velocity of objects classified as hazardous vs non hazardous:', median_hazardous_velocity - median_non_hazardous_velocity)

print('\n')
#there seems to be a noticable difference between velocities of objects classified as hazardous vs non hazardous

sns.boxplot(data=neo, x='hazardous', y='relative_velocity')
plt.xlabel('Hazardous')
plt.ylabel('Relative Velocity (kms/hr)')
plt.title('Relationship Between Object Hazardousness and Relative Velocity')
# plt.show()
plt.clf()
#boxplot shows only partial overlap between the two variables, further indicating variance between the two.

plt.hist(true_hazardous, color='blue', label='True', density=True, alpha=.5)
plt.hist(false_hazardous, color='red', label='False', density=True, alpha=.5)
plt.legend()
plt.xlabel('Hazardous')
plt.ylabel('Relative Velocity (kms/hr)')
plt.title('Relationship between Object Hazardousness and Relative Velocity')
# plt.show()
plt.clf()
#histogram indicates same as boxplot


#analyzing relationship between est_diameter_min and relative_velocity

#limit velocity and diameter and select random subset to make visualization easier to interpret
#I looked to the trim means for both variables as a guide for how much to limit the results of my graph.

random_subset = neo.sample(n=1000)

plt.scatter(x=random_subset.relative_velocity, y=random_subset.est_diameter_min)
# plt.xlim(0, 75000)
# plt.ylim(0, .15)
plt.xlabel('Relative Velocity (km/hr)')
plt.ylabel('Estimated Minimum Diameter (km)')
plt.title('Relationship Between Relative Velocity and Estimated Minimum Diameter')
plt.savefig('neo_velocity_diameter.png')

predicted_velocity = 18773 * neo.est_diameter_min + 45675
plt.plot(neo.est_diameter_min, predicted_velocity)
# plt.show()
plt.clf()
#results are hard for me to interpret, I will next try linear regression to further explore the relationship between the two variables.

velocity_diameter_model = sm.OLS.from_formula('relative_velocity ~ est_diameter_min', data=neo)
model_results = velocity_diameter_model.fit()
print('model results to predict velocity from diameter:\n', np.round(model_results.params))

#relative_velocity = 18773 * est_diameter_min + 45675

fitted_values = model_results.predict(neo)
residuals = neo.relative_velocity - fitted_values

#verify that the residuals are normally distributed
plt.hist(residuals, bins=50)
plt.xlim(-100000, 100000)
# plt.show()
plt.clf()

#verify that the residuals have equal variation
plt.scatter(fitted_values, residuals)
plt.xlim(40000, 100000)
plt.ylim(-100000, 150000)
# plt.show()
plt.clf()
#there does not appear to be any patterns or asymmetry to the data.

#find covariance and correlation between relative velocity and minimum estimated diameter
cov_mat_velocity_diameter = np.cov(neo.relative_velocity, neo.est_diameter_min)
print("covariance matrix for relative velocity and minimum estimated diameter:\n", cov_mat_velocity_diameter)

corr_velocity_diameter, p = pearsonr(neo.relative_velocity, neo.est_diameter_min)
print('correlation between relative velocity and minimum estimated diameter:\n', np.round(corr_velocity_diameter, 2))
#there is a positive correlation of .22 indicating a weak, insignificant linear association.