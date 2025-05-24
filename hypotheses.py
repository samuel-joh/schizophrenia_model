import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir(os.path.join(os.getcwd(), "data_for_modeling", "extracted_timeseries_schizophrenia"))

subfolders = [os.path.join(f.path, "func") for f in os.scandir(os.getcwd()) if f.is_dir()]

data = []
for folder in subfolders:
    for file in os.scandir(folder):
        with open(file, "r") as f:
            data.append(pd.read_csv(file, sep='\t'))

lh_vision_columns = [col for col in data[0].columns if col.startswith('LH_Vis') or col.startswith('RH_Vis')]

# Show the columns
print("Columns starting with 'Vis':")
print(lh_vision_columns)

data_control = data[:22]
data_schizo = data[-23:]

# If you want to plot these columns
plt.figure(figsize=(15, 10))
label = 'Vision (control)'
labeled = False
for col in lh_vision_columns:
    for tms in data_control:
        plt.plot(tms.index, tms[col], color="blue", label=label)  # Plot each column starting with 'LH_Vis' or 'RH_Vis'
        label = "_nolegend_"
    label = "Vision (schizo)"
    for tms in data_schizo:
        if labeled:
            label = "_nolegend_"
        plt.plot(tms.index, tms[col], color="red", label=label)  # Plot each column starting with 'LH_Vis' or 'RH_Vis'
        labeled = True

print(f"Data from {len(data_control)} healthy people and {len(data_schizo)} schizophrenia patients")
plt.xlabel('Time (timepoints)')
plt.ylabel('BOLD signal')
plt.title(f'Time Series for Vision regions \n{lh_vision_columns[:2]}, ..., {lh_vision_columns[-2:]}')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside the plot
plt.tight_layout()
os.chdir(os.path.join(os.getcwd(), "..", ".."))
plt.savefig("Striatum.png")
plt.show()

