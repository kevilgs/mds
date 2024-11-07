import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import RandomOverSampler
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load datasets
logger.info("Loading datasets...")
df_comb = pd.read_csv('data/dis_sym_dataset_comb.csv')
df_norm = pd.read_csv('data/dis_sym_dataset_norm.csv')

# Prepare df_comb for merging
logger.info("Preparing df_comb for merging...")
df_comb_melted = df_comb.melt(id_vars=['label_dis'], var_name='Symptom', value_name='Present')
df_comb_filtered = df_comb_melted[df_comb_melted['Present'] == 1].copy()
df_comb_filtered['Disease'] = df_comb_filtered['label_dis']

# Prepare df_norm for merging
logger.info("Preparing df_norm for merging...")
df_norm_melted = df_norm.melt(id_vars=['label_dis'], var_name='Symptom', value_name='Present')
df_norm_filtered = df_norm_melted[df_norm_melted['Present'] == 1].copy()
df_norm_filtered['Disease'] = df_norm_filtered['label_dis']

# Combine both datasets
logger.info("Combining datasets...")
combined_df = pd.concat([df_comb_filtered[['Symptom', 'Disease']], df_norm_filtered[['Symptom', 'Disease']]], ignore_index=True)

# Check class distribution
class_distribution = combined_df['Disease'].value_counts()
logger.info("Class distribution:\n%s", class_distribution)

# Filter out classes with fewer than a certain number of samples
min_samples = 5
classes_to_keep = class_distribution[class_distribution >= min_samples].index
filtered_df = combined_df[combined_df['Disease'].isin(classes_to_keep)].copy()

# Reduce dataset size for debugging (optional)
filtered_df = filtered_df.sample(frac=0.1, random_state=42)  # Use 10% of the data

# Encode symptoms and diseases
symptom_encoder = LabelEncoder()
disease_encoder = LabelEncoder()

# Fit the symptom encoder with all possible symptoms
all_symptoms = combined_df['Symptom'].unique()
symptom_encoder.fit(all_symptoms)

filtered_df['Symptom_Encoded'] = symptom_encoder.transform(filtered_df['Symptom'])
filtered_df['Disease_Encoded'] = disease_encoder.fit_transform(filtered_df['Disease'])

# Add additional features (for demonstration purposes, randomly generated)
np.random.seed(42)
filtered_df['Overweight'] = np.random.choice(['yes', 'no', 'dont_know'], size=len(filtered_df))
filtered_df['Smoke'] = np.random.choice(['yes', 'no', 'dont_know'], size=len(filtered_df))
filtered_df['Injured'] = np.random.choice(['yes', 'no', 'dont_know'], size=len(filtered_df))
filtered_df['Cholesterol'] = np.random.choice(['yes', 'no', 'dont_know'], size=len(filtered_df))
filtered_df['Hypertension'] = np.random.choice(['yes', 'no', 'dont_know'], size=len(filtered_df))
filtered_df['Diabetes'] = np.random.choice(['yes', 'no', 'dont_know'], size=len(filtered_df))

# Encode additional features
feature_mapping = {'yes': 1, 'no': 0, 'dont_know': 2}
filtered_df['Overweight'] = filtered_df['Overweight'].map(feature_mapping)
filtered_df['Smoke'] = filtered_df['Smoke'].map(feature_mapping)
filtered_df['Injured'] = filtered_df['Injured'].map(feature_mapping)
filtered_df['Cholesterol'] = filtered_df['Cholesterol'].map(feature_mapping)
filtered_df['Hypertension'] = filtered_df['Hypertension'].map(feature_mapping)
filtered_df['Diabetes'] = filtered_df['Diabetes'].map(feature_mapping)

# Create feature matrix X and target vector y
X = pd.get_dummies(filtered_df['Symptom_Encoded'], drop_first=True)
X['Overweight'] = filtered_df['Overweight']
X['Smoke'] = filtered_df['Smoke']
X['Injured'] = filtered_df['Injured']
X['Cholesterol'] = filtered_df['Cholesterol']
X['Hypertension'] = filtered_df['Hypertension']
X['Diabetes'] = filtered_df['Diabetes']

# Convert all column names to strings
X.columns = X.columns.astype(str)

y = filtered_df['Disease_Encoded']

# Save the column names
columns = X.columns.tolist()
with open('model/columns.pkl', 'wb') as f:
    joblib.dump(columns, f)

# Perform feature selection
logger.info("Performing feature selection...")
selector = SelectKBest(chi2, k=min(500, X.shape[1]))  # Select top 1000 features or all if less
X_selected = selector.fit_transform(X, y)

# Data augmentation with RandomOverSampler
logger.info("Performing data augmentation with RandomOverSampler...")
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_selected, y)

# Manually set hyperparameters
logger.info("Setting hyperparameters manually...")
log_model = LogisticRegression(C=1, solver='liblinear', max_iter=200, random_state=42)

# Cross-validation
logger.info("Performing cross-validation...")
cv_scores = cross_val_score(log_model, X_resampled, y_resampled, cv=5, scoring='accuracy')
logger.info("Cross-validation scores: %s", cv_scores)
logger.info("Mean cross-validation score: %s", cv_scores.mean())

# Splitting the dataset into training and testing sets
logger.info("Splitting the dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=42)

# Train the Logistic Regression model
logger.info("Training the Logistic Regression model...")
log_model.fit(X_train, y_train)

# Save the model
logger.info("Saving the model...")
joblib.dump(log_model, 'model/logistic_regression_model.pkl')
joblib.dump(symptom_encoder, 'model/symptom_encoder.pkl')  # Save the symptom encoder
joblib.dump(disease_encoder, 'model/disease_encoder.pkl')  # Save the disease encoder

logger.info("Model training complete and saved!")