import cdflib
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# === SETTINGS ===
input_path = 'data/AL1_ASW91_L2_TH2_20250630_UNP_9999_999999_V02.cdf'
output_dir = 'cdf_json_output'
# variable_name = 'Proton_Density'  # Change this to any other variable if needed

# === PREPARE OUTPUT DIR ===
os.makedirs(output_dir, exist_ok=True)

# === LOAD CDF FILE ===
cdf_file = cdflib.CDF(input_path)

# === EXTRACT & SAVE METADATA ===
info = cdf_file.cdf_info()

# Create a simplified metadata dictionary
clean_metadata = {
    "CDF Path": str(info.CDF),
    "Version": info.Version,
    "Encoding": info.Encoding,
    "Majority": info.Majority,
    "rVariables": info.rVariables,
    "zVariables": info.zVariables,
    "Attributes": {},  # We'll convert list of dicts to flat dict
    "Compressed": info.Compressed,
    "Checksum": info.Checksum,
    "LeapSecondUpdate": str(info.LeapSecondUpdate),
    "Copyright": info.Copyright.strip(),
}

# Convert list of dicts (Attributes) into a flat dict
for item in info.Attributes:
    for k, v in item.items():
        clean_metadata["Attributes"][k] = v

# Save to JSON
with open(f"{output_dir}/metadata_clean.json", "w") as f:
    json.dump(clean_metadata, f, indent=4)

print(f"[✓] Clean metadata saved to metadata_clean.json")