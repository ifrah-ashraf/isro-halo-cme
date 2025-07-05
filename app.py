import cdflib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# === 1. Load the CDF file ===
cdf_file = cdflib.CDF('data/data.cdf')

# === 2. Extract variables ===
epoch = cdf_file.varget('epoch_for_cdf_mod')
flux = cdf_file.varget('integrated_flux_mod')
energy = cdf_file.varget('energy_center_mod')
flux_uncer = cdf_file.varget('flux_uncer')
xpos = cdf_file.varget('spacecraft_xpos')
ypos = cdf_file.varget('spacecraft_ypos')
zpos = cdf_file.varget('spacecraft_zpos')
sun_angle = cdf_file.varget('sun_angle_tha2')

# === 3. Convert epoch to datetime ===
datetimes = pd.to_datetime(cdflib.cdfepoch.to_datetime(epoch))

# === 4. Feature engineering helpers ===
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

# === 5. Build features DataFrame ===
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

# === 6. Label current CME events (as before) ===
features['flux_diff'] = features['flux_mean'].diff()
THRESHOLD = 1e7
cme_arrival_indices = features.index[features['flux_diff'] > THRESHOLD].tolist()
cme_arrival_times = features.loc[cme_arrival_indices, 'datetime'].tolist()
label_window_hours = 3
features['label'] = 0
for arrival_time in cme_arrival_times:
    window_start = arrival_time - pd.Timedelta(hours=label_window_hours)
    window_end = arrival_time + pd.Timedelta(hours=label_window_hours)
    features.loc[(features['datetime'] >= window_start) & (features['datetime'] <= window_end), 'label'] = 1

# === 7. Forecasting: Create lagged features and future CME label ===
# Define how many hours in the past and future to use
history_hours = 24  # Use past 24 hours as input
future_window = 24   # Predict CME in next 24 hours

# Assume data is hourly; adjust if your cadence is different
for lag in range(1, history_hours + 1):
    for col in ['flux_mean', 'flux_max', 'flux_min', 'energy_mean', 'energy_max', 'energy_min', 'sun_angle_mean', 'sun_angle_std']:
        features[f'{col}_lag_{lag}'] = features[col].shift(lag)

# Label: 1 if a CME occurs in the next 24 hours
features['future_cme'] = features['label'].rolling(window=future_window, min_periods=1).max().shift(-future_window)
features['future_cme'] = features['future_cme'].fillna(0).astype(int)

# Drop rows with NaN due to shifting
features = features.dropna().reset_index(drop=True)

# === 8. Train/test split and ML ===
lagged_feature_cols = [col for col in features.columns if '_lag_' in col]
X = features[lagged_feature_cols]
y = features['future_cme']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42, test_size=0.33
)
clf = RandomForestClassifier(random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# === 9. Evaluate ===
y_pred = clf.predict(X_test)
print("Classification report for future CME prediction:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === 10. Example: Predicting CME for a new time (using past 24 hours) ===
# Let's pick the most recent row in the features DataFrame
new_X = X.iloc[[-1]]
future_prediction = clf.predict(new_X)[0]
print(f"Prediction for the next {future_window} hours: {'CME likely' if future_prediction else 'No CME expected'}")

# === 11. Plotting: Label and prediction distribution ===
plt.figure(figsize=(12, 4))
plt.plot(features['datetime'], features['future_cme'], label='Future CME label')
plt.xlabel('Time')
plt.ylabel(f'CME in next {future_window}h')
plt.title('CME Forecast Labels Over Time')
plt.legend()
plt.show()
