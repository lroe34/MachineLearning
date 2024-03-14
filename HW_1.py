import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
 
scaler = StandardScaler()

def remove_labels(df, column):
    features = df.drop(column, axis=1)
    labels = df[column]

    return features, labels


def scale_data(df):
    scaled_df = pd.DataFrame(scaler.fit_transform(df))
    return scaled_df

def min_max(df):
    min_max_scaler = MinMaxScaler()
    min_max_df = pd.DataFrame(min_max_scaler.fit_transform(df))
    return min_max_df

def VCR(data):
    U, s, V = np.linalg.svd(data)
    vcr = s[0] / np.sum(s)

    return vcr



df1 = pd.read_csv('breast_wisc_dataset.csv')
df2 = pd.read_csv('cybersecurity_data.csv')
df3 = pd.read_csv('HFT_AAPL_data.csv')
df4 = pd.read_csv('music_data.csv')
df5 = pd.read_csv('OptionDataOTM_data.csv')
df6 = pd.DataFrame(load_iris().data)

df1_features, df1_labels = remove_labels(df1, 'label')
df2_features, df2_labels = remove_labels(df2, 'class')
df3_features, df3_labels = remove_labels(df3, 'Date')
df4_features, df4_labels = remove_labels(df4, 'class')

print(df1_features.shape)
print(df2_features.shape)
print(df3_features.shape)
print(df4_features.shape)
print(df5.shape)
print(df6.shape)


df1_scaled = scale_data(df1_features)
df2_scaled = scale_data(df2_features)
df3_scaled = scale_data(df3_features)
df4_scaled = scale_data(df4_features)
df5_scaled = scale_data(df5)
df6_scaled = scale_data(df6)



# print('VCR for SS df1:', VCR(df1_scaled))
# print('VCR for SS df2:', VCR(df2_scaled))
# print('VCR for SS df3:', VCR(df3_scaled))
# print('VCR for SS df4:', VCR(df4_scaled))
# print('VCR for SS df5:', VCR(df5_scaled))
# print('VCR for SS df6:', VCR(df6_scaled))

df1_min_max = min_max(df1_features)
df2_min_max = min_max(df2_features)
df3_min_max = min_max(df3_features)
df4_min_max = min_max(df4_features)
df5_min_max = min_max(df5)
df6_min_max = min_max(df6)

# print('VCR for MM df1:', VCR(df1_min_max))
# print('VCR for MM df2:', VCR(df2_min_max))
# print('VCR for MM df3:', VCR(df3_min_max))
# print('VCR for MM df4:', VCR(df4_min_max))
# print('VCR for MM df5:', VCR(df5_min_max))
# print('VCR for MM df6:', VCR(df6_min_max))

# print('VCR for df1:', VCR(df1_features))
# print('VCR for df2:', VCR(df2_features))
# print('VCR for df3:', VCR(df3_features))
# print('VCR for df4:', VCR(df4_features))
# print('VCR for df5:', VCR(df5))
# print('VCR for df6:', VCR(df6))



vcr_values = {
    'SS': [VCR(df1_scaled), VCR(df2_scaled), VCR(df3_scaled), VCR(df4_scaled), VCR(df5_scaled), VCR(df6_scaled)],
    'MM': [VCR(df1_min_max), VCR(df2_min_max), VCR(df3_min_max), VCR(df4_min_max), VCR(df5_min_max), VCR(df6_min_max)],
    'Original': [VCR(df1_features), VCR(df2_features), VCR(df3_features), VCR(df4_features), VCR(df5), VCR(df6)]
}
print(vcr_values)
datasets = ['df1', 'df2', 'df3', 'df4', 'df5', 'df6']

plt.figure(figsize=(10, 8))

n_datasets = len(datasets)
ind = np.arange(n_datasets)  
width = 0.25       

plt.bar(ind, vcr_values['SS'], width, label='Standard Scaler')
plt.bar(ind + width, vcr_values['MM'], width, label='Min-Max Scaler')
plt.bar(ind + 2*width, vcr_values['Original'], width, label='Original')

plt.ylabel('VCR Values')
plt.title('VCR by dataset and scaling method')
plt.xticks(ind + width, datasets)
plt.legend(loc='best')

plt.show()

def plot_vcr_for_dataset(dataset_name, vcr_values):
    plt.figure(figsize=(6, 4))
    methods = ['Standard Scaler', 'Min-Max Scaler', 'Original']
    values = [vcr_values['SS'][datasets.index(dataset_name)], 
              vcr_values['MM'][datasets.index(dataset_name)], 
              vcr_values['Original'][datasets.index(dataset_name)]]
    
    plt.bar(methods, values, color=['blue', 'orange', 'green'])
    plt.ylabel('VCR Values')
    plt.title(f'VCR for {dataset_name} by Scaling Method')
    plt.savefig(f'{dataset_name}_VCR_comparison.png')
    plt.close()

for dataset in datasets:
    plot_vcr_for_dataset(dataset, vcr_values)