import cdflib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# === 1. Load the CDF file ===
cdf_file = cdflib.CDF('data\data.cdf')

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
datetimes = cdflib.cdfepoch.to_datetime(epoch)

# === 4. Print shapes for debugging ===
print('datetimes:', np.shape(datetimes))
print('flux:', np.shape(flux))
print('energy:', np.shape(energy))
print('flux_uncer:', np.shape(flux_uncer))
print('xpos:', np.shape(xpos))
print('ypos:', np.shape(ypos))
print('zpos:', np.shape(zpos))
print('sun_angle:', np.shape(sun_angle))

# === 5. Feature engineering helpers ===
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
# Truncate all arrays to the same length

def truncate(arr):
    arr = np.asarray(arr)
    if arr.ndim == 0:
        # Scalar: repeat to match length
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

# === 6. Build features DataFrame ===
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

print(features.head())

# === 7. AUTOMATED CME ARRIVAL DETECTION FROM IN-SITU DATA ===

# 1. Compute the difference in flux_mean between consecutive time steps
features['flux_diff'] = features['flux_mean'].diff()

# 2. Choose a threshold for a significant jump (adjust as needed)
THRESHOLD = 5e6  # You may need to tune this value for your dataset

# 3. Find indices where the difference exceeds the threshold
cme_arrival_indices = features.index[features['flux_diff'] > THRESHOLD].tolist()

# 4. Get the corresponding arrival times
cme_arrival_times = features.loc[cme_arrival_indices, 'datetime'].tolist()
print("Detected CME arrivals at:")
for t in cme_arrival_times:
    print(t)

# 5. Label as CME within a window around each detected arrival
label_window_hours = 12  # Label +/- 12 hours around each detected arrival
features['label'] = 0  # default: no CME

for arrival_time in cme_arrival_times:
    window_start = arrival_time - pd.Timedelta(hours=label_window_hours)
    window_end = arrival_time + pd.Timedelta(hours=label_window_hours)
    features.loc[(features['datetime'] >= window_start) & (features['datetime'] <= window_end), 'label'] = 1

print(features['label'].value_counts())


# === 9. Train/test split and ML ===
X = features.drop(['datetime', 'label'], axis=1)
y = features['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# === 10. Evaluate ===
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# === 11. Predict on new data point ===
new_point = {
    'xpos': 1.2e6,
    'ypos': 2e5,
    'zpos': -1e5,
    'flux_mean': 1.5e6,
    'flux_max': 2e6,
    'flux_min': 1e5,
    'energy_mean': 3500,
    'energy_max': 4000,
    'energy_min': 3000,
    'sun_angle_mean': 0.1,
    'sun_angle_std': 0.05
}
test_time = pd.Timestamp('2025/06/29 16:12')
# Find the index of the closest datetime
i_closest = (features['datetime'] - test_time).abs().idxmin()
row = features.loc[i_closest]

new_point = row.drop(['datetime', 'label']).to_dict()
print("CME detected" if clf.predict(pd.DataFrame([new_point]))[0] else "No CME detected")


mask = (features['datetime'] > test_time - pd.Timedelta('2D')) & (features['datetime'] < test_time + pd.Timedelta('2D'))
features[mask].plot(x='datetime', y='flux_mean')
plt.axvline(test_time, color='red', linestyle='--')
plt.show()