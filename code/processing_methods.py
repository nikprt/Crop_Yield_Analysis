import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Method for reading in a csv file:
def read_data(path):
    data = pd.read_excel(path, names=['rel_yield_loss', 'airTemp_mean_spring', 'airTemp_mean_summer',
                                    'hot_days_average', 'frost_days_average', 'precip_mean_spring',
                                    'precip_mean_summer', 'sunDur_mean_spring', 'sunDur_mean_summer'])

    return data

# Method for plotting the hot days averages of each state:
def plot_hot_days(data):

    m = np.linspace(1999,2019,21).astype(int)

    fig = plt.figure(figsize=(12, 5))

    plt.plot(m, data.iloc[0:21,3], label='SH')
    plt.plot(m, data.iloc[21:42,3], label='NI')
    plt.plot(m, data.iloc[42:63,3], label='NRW')
    plt.plot(m, data.iloc[63:84,3], label='HE')
    plt.plot(m, data.iloc[84:105,3], label='RLP')
    plt.plot(m, data.iloc[105:126,3], label='BW')
    plt.plot(m, data.iloc[126:147,3], label='BY')
    plt.plot(m, data.iloc[147:168,3], label='SL')
    plt.plot(m, data.iloc[168:189,3], label='BB', c='indianred')
    plt.plot(m, data.iloc[189:210,3], label='MP', c='darkblue')
    plt.plot(m, data.iloc[210:231,3], label='SAC', c='teal')
    plt.plot(m, data.iloc[231:252,3], label='SA', c='darkred')
    plt.plot(m, data.iloc[252:273,3], label='THU', c='peru')

    # Set the year as xticklabels:
    plt.xticks(m, rotation=90, fontweight='bold')
    plt.ylabel('Hot Days (>29°C)', fontweight='bold')
    plt.title('Hot Days States Averages (1999-2019)', fontsize=14, fontweight='bold')
    plt.grid(axis='x')
    plt.legend(loc='upper left')

    plt.show()

# Method for plotting the mean precipitation values of each state:
def plot_precipitation_means(data):
    precip_data = {'SH': data.iloc[0:21,6].values,
                   'NI': data.iloc[21:42,6].values,
                   'NRW': data.iloc[42:63,6].values,
                   'HE': data.iloc[63:84,6].values,
                   'RLP': data.iloc[84:105,6].values,
                   'BW': data.iloc[105:126,6].values,
                   'BY': data.iloc[126:147,6].values,
                   'SL': data.iloc[147:168,6].values,
                   'BB': data.iloc[168:189,6].values,
                   'MP': data.iloc[189:210,6].values,
                   'SAC': data.iloc[210:231,6].values,
                   'SA': data.iloc[231:252,6].values,
                   'THU': data.iloc[252:273,6].values}

    colors=['cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue',
            'cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue',
            'cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue',
            'cornflowerblue']

    fig, ax = plt.subplots(figsize=(12,5))
    boxplot = ax.boxplot(precip_data.values(), vert=True, patch_artist=True, labels=precip_data.keys())
    ax.set_xticklabels(precip_data.keys(), fontweight='bold')
    ax.xaxis.grid()
    ax.set_ylabel('Average Precipitation (mm)', fontweight='bold')
    #ax.set_title('All Scenarios (Brick & Mortar)')

    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Mean Summer Precipitation (1999-2019)', fontsize=14, fontweight='bold')
    plt.show()

# Method for plotting the yield clusters of testset:
def plot_testset_clusters(df_test, X_val, y_pred_val, km):

    # Initialize clusters as dataframes:
    df1_test = df_test[df_test.cluster == 0]
    df2_test = df_test[df_test.cluster == 1]
    df3_test = df_test[df_test.cluster == 2]
    df4_test = df_test[df_test.cluster == 3]
    df5_test = df_test[df_test.cluster == 4]

    # Plot the clusters in 3D-Scatter-Plot:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(df1_test.airTemp_mean_summer, df1_test.hot_days_average, df1_test['rel_yield_loss'], color='blue')
    ax.scatter(df2_test.airTemp_mean_summer, df2_test.hot_days_average, df2_test['rel_yield_loss'], color='cornflowerblue')
    ax.scatter(df3_test.airTemp_mean_summer, df3_test.hot_days_average, df3_test['rel_yield_loss'], color='darkcyan')
    ax.scatter(df4_test.airTemp_mean_summer, df4_test.hot_days_average, df4_test['rel_yield_loss'], color='lightslategray')
    ax.scatter(df5_test.airTemp_mean_summer, df5_test.hot_days_average, df5_test['rel_yield_loss'], color='purple')
    ax.scatter(X_val[:, 0], X_val[:, 1], y_pred_val, s=25, c='red', marker='o', label='model prediction')  # predicted rel crop yield for input data
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], km.cluster_centers_[:, 2], color='gold', marker='+', s=50, label='cluster centroids')
    ax.set_xlabel('Mean Airtemp. Summer (°C)', fontweight='bold')
    ax.set_ylabel('Average Hot Days (d)', fontweight='bold')
    ax.set_zlabel('rel. crop yield (%)', fontweight='bold')
    plt.title('Testset Crop Yield Clusters', fontsize=14, fontweight='bold')
    plt.legend()
    plt.show()
