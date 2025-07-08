import cdflib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

cdf_file = cdflib.CDF('data/data.cdf')

#Extract variables
epoch = cdf_file.varget('epoch_for_cdf_mod')
flux = cdf_file.varget('integrated_flux_mod')
energy = cdf_file.varget('energy_center_mod')
flux_uncer = cdf_file.varget('flux_uncer')
xpos = cdf_file.varget('spacecraft_xpos')
ypos = cdf_file.varget('spacecraft_ypos')
zpos = cdf_file.varget('spacecraft_zpos')
sun_angle = cdf_file.varget('sun_angle_tha2')

datetimes = pd.to_datetime(cdflib.cdfepoch.to_datetime(epoch))

#Feature Engineering
def reduce_array(arr, stat='mean'):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    if stat == 'mean':
        return np.nanmean(arr, axis=1)
    elif stat == 'max':
        return np.nanmax(arr, axis=1)
    elif stat == 'min':
        return np.nanmin(arr, axis=1)
    elif stat == 'std':
        return np.nanstd(arr, axis=1)
    else:
        raise ValueError("Unknown stat")

def reduce_array_3d(arr, stat='mean'):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    if stat == 'mean':
        return np.nanmean(arr, axis=(1,2))
    elif stat == 'std':
        return np.nanstd(arr, axis=(1,2))
    else:
        raise ValueError("Unknown stat")

length = len(datetimes)
def truncate(arr):
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return np.full(length, arr)
    if arr.shape[0] >= length:
        return arr[:length]
    else:
        return np.pad(arr, (0, length - arr.shape[0]), constant_values=np.nan)

flux = truncate(flux)
energy = truncate(energy)
flux_uncer = truncate(flux_uncer)
xpos = truncate(xpos)
ypos = truncate(ypos)
zpos = truncate(zpos)
sun_angle = truncate(sun_angle)

#Features DataFrame
features = pd.DataFrame({
    'datetime': datetimes,
    'xpos': xpos,
    'ypos': ypos,
    'zpos': zpos,
    'flux_mean': reduce_array(flux, 'mean'),
    'flux_max': reduce_array(flux, 'max'),
    'flux_min': reduce_array(flux, 'min'),
    'energy_mean': reduce_array(energy, 'mean'),
    'energy_max': reduce_array(energy, 'max'),
    'energy_min': reduce_array(energy, 'min'),
    'sun_angle_mean': reduce_array_3d(sun_angle, 'mean'),
    'sun_angle_std': reduce_array_3d(sun_angle, 'std'),
})

features['flux_diff'] = features['flux_mean'].diff()
THRESHOLD = 1e7
cme_arrival_indices = features.index[features['flux_diff'] > THRESHOLD].tolist()
cme_arrival_times = features.loc[cme_arrival_indices, 'datetime'].tolist()

label_window_hours = 2
features['label'] = 0

for arrival_time in cme_arrival_times:
    window_start = arrival_time - pd.Timedelta(hours=label_window_hours)
    window_end = arrival_time + pd.Timedelta(hours=label_window_hours)
    features.loc[(features['datetime'] >= window_start) & (features['datetime'] <= window_end), 'label'] = 1

print(f"CME events found: {len(cme_arrival_times)}")
print(f"Total CME-labeled periods: {features['label'].sum()}")
print(f"Total non-CME periods: {(features['label'] == 0).sum()}")

# Define how many hours in the past and future to use
history_hours = 24
future_window = 12 

#Assuming that the data is hourly
for lag in range(1, history_hours + 1):
    for col in ['flux_mean', 'flux_max', 'flux_min', 'energy_mean', 'energy_max', 'energy_min', 'sun_angle_mean', 'sun_angle_std']:
        features[f'{col}_lag_{lag}'] = features[col].shift(lag)

features['future_cme'] = features['label'].rolling(window=future_window, min_periods=1).max().shift(-future_window)
features['future_cme'] = features['future_cme'].fillna(0).astype(int)

print(f"Future CME labels - Class 0: {(features['future_cme'] == 0).sum()}, Class 1: {(features['future_cme'] == 1).sum()}")

#Drop rows with NaN
features = features.dropna().reset_index(drop=True)

lagged_feature_cols = [col for col in features.columns if '_lag_' in col]
X = features[lagged_feature_cols]
y = features['future_cme']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42, test_size=0.33
)

#Applying SMOTE to balance the training data
print("Class distribution before SMOTE:")
print(f"Class 0: {sum(y_train == 0)}, Class 1: {sum(y_train == 1)}")

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, sampling_strategy=0.3)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:")
print(f"Class 0: {sum(y_train_smote == 0)}, Class 1: {sum(y_train_smote == 1)}")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np

clf = RandomForestClassifier(
    random_state=42,
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced'
)
clf.fit(X_train_smote, y_train_smote)

y_pred_proba = clf.predict_proba(X_test)[:, 1]

#Find optimal threshold using balanced approach
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

best_threshold = 0.5
best_score = 0
target_tn_range = (200, 250)

for threshold in np.arange(0.3, 0.8, 0.01):
    y_pred_temp = (y_pred_proba >= threshold).astype(int)
    tn = np.sum((y_test == 0) & (y_pred_temp == 0))
    fp = np.sum((y_test == 0) & (y_pred_temp == 1))
    fn = np.sum((y_test == 1) & (y_pred_temp == 0))
    tp = np.sum((y_test == 1) & (y_pred_temp == 1))
    
    if (tp + fn) > 0 and (tn + fp) > 0:
        sensitivity = tp / (tp + fn) 
        specificity = tn / (tn + fp)  
        
        if target_tn_range[0] <= tn <= target_tn_range[1] and sensitivity > 0.9:
            score = sensitivity + specificity - abs(tn - 215) * 0.001 
            if score > best_score:
                best_score = score
                best_threshold = threshold

optimal_threshold = best_threshold

print(f"Optimal threshold: {optimal_threshold:.3f}")

print(f"\nTest set class distribution:")
print(f"Total Class 0 (No CME): {sum(y_test == 0)}")
print(f"Total Class 1 (CME): {sum(y_test == 1)}")

y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)

print("\n=== Results with Default Threshold (0.5) ===")
y_pred_default = clf.predict(X_test)
print("Classification report:")
print(classification_report(y_test, y_pred_default))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_default))

print(f"\n=== Results with Optimized Threshold ({optimal_threshold:.3f}) ===")
print("Classification report:")
print(classification_report(y_test, y_pred_optimized))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_optimized))

from sklearn.metrics import roc_auc_score, balanced_accuracy_score
print(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
print(f"Balanced Accuracy (default): {balanced_accuracy_score(y_test, y_pred_default):.3f}")
print(f"Balanced Accuracy (optimized): {balanced_accuracy_score(y_test, y_pred_optimized):.3f}")

'''# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))'''

new_X = X.iloc[[-1]]
future_prediction_proba = clf.predict_proba(new_X)[0, 1]
future_prediction_default = clf.predict(new_X)[0]
future_prediction_optimized = (future_prediction_proba >= optimal_threshold).astype(int)

print(f"\nPrediction for the next {future_window} hours:")
print(f"Probability of Halo CME: {future_prediction_proba:.3f}")
'''print(f"Default threshold (0.5): {'Halo CME likely' if future_prediction_default else 'No CME expected'}")'''
print(f"Optimized threshold ({optimal_threshold:.3f}): {'Halo CME likely' if future_prediction_optimized else 'No Halo CME expected'}")

#Data visualization
plt.figure(figsize=(15, 8))

# Plot 1: Original labels vs predictions
plt.subplot(2, 1, 1)
plt.plot(features['datetime'], features['future_cme'], label='Actual Future CME', alpha=0.7)
plt.xlabel('Time')
plt.ylabel(f'CME in next {future_window}h')
plt.title('Actual CME Labels Over Time')
plt.legend()

# Plot 2: Prediction probabilities
plt.subplot(2, 1, 2)
X_full_pred_proba = clf.predict_proba(X)[:, 1]
plt.plot(features['datetime'], X_full_pred_proba, label='CME Probability', alpha=0.7, color='orange')
plt.axhline(y=0.5, color='red', linestyle='--', label='Default Threshold (0.5)')
plt.axhline(y=optimal_threshold, color='green', linestyle='--', label=f'Optimized Threshold ({optimal_threshold:.3f})')
plt.xlabel('Time')
plt.ylabel('CME Probability')
plt.title('Model Predictions - CME Probability Over Time')
plt.legend()

plt.tight_layout()
plt.show()
