import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score


def cat_feature_analysis_horizontal(data, var, missing_class=False):
    df = data.copy()

    # summary statistics
    print('Summary statistics: ')
    print(df[var].describe())
    print()

    # missing share
    missing_share = df[var].isnull().sum() / df.shape[0] * 100
    print(f"Percentage of missing data: {missing_share}%")
    print()

    # conversion rate on missing vs non-missing
    if missing_share > 0:
        df['is_missing'] = df[var].isnull().astype(int)
        miss_rate = df.groupby('is_missing')['has_subscribed'].mean()
        print("Conversion rate - missing data: ", miss_rate[1].round(2))
        print()
        print("Conversion rate - non-missing: ", miss_rate[0].round(2))
        print()

    if missing_class:
        # create a missing class
        df.loc[df[var].isnull(), var] = 'missing'

    print(f"Percentage of each discrete values: ")
    print((df[var].value_counts(normalize=True) * 100).round(1))
    print()

    nb_cat = df[var].nunique()

    fig, axs = plt.subplots(3, 1, figsize=(min(nb_cat, 15), 9))

    index_order = sorted(df[var].value_counts().index)

    # Count plot
    axs[0].set_title('Lead count per class', fontsize=11, loc='left')
    sns.countplot(data=df, x=var, ax=axs[0], order=index_order)

    # Count plot per default class
    axs[1].set_title('Conversion count per class', fontsize=11, loc='left')
    sns.countplot(data=df, x=var, hue='has_subscribed', ax=axs[1], order=index_order)
    axs[1].legend(loc='best')

    # average default rate per bins
    axs[2].set_title('Conversion rate per class', fontsize=11, loc='left')
    sns.barplot(data=df, x=var, y='has_subscribed', errorbar=None, ax=axs[2], order=index_order)

    for ax in axs.flat:
        ax.set(xlabel='', ylabel='')
    fig.tight_layout()


def cat_feature_analysis(data, var, missing_class=False):
    df = data.copy()

    # summary statistics
    print('Summary statistics: ')
    print(df[var].describe())
    print()

    # missing share
    missing_share = df[var].isnull().sum() / df.shape[0] * 100
    print(f"Percentage of missing data: {missing_share}%")
    print()

    # conversion rate on missing vs non-missing
    if missing_share > 0:
        df['is_missing'] = df[var].isnull().astype(int)
        miss_rate = df.groupby('is_missing')['has_subscribed'].mean()
        print("Conversion rate - missing data: ", miss_rate[1].round(2))
        print()
        print("Conversion rate - non-missing: ", miss_rate[0].round(2))
        print()

    if missing_class:
        # create a missing class
        df.loc[df[var].isnull(), var] = 'missing'

    print(f"Percentage of each discrete values: ")
    print((df[var].value_counts(normalize=True) * 100).round(1))
    print()

    nb_cat = df[var].nunique()

    fig, axs = plt.subplots(1, 3, figsize=(15, max(nb_cat / 2, 3)))

    index_order = sorted(df[var].value_counts().index)

    # Count plot
    axs[0].set_title('Lead count per class', fontsize=11, loc='left')
    sns.countplot(data=df, y=var, ax=axs[0], order=index_order)

    # Count plot per default class
    axs[1].set_title('Conversion count per class', fontsize=11, loc='left')
    sns.countplot(data=df, y=var, hue='has_subscribed', ax=axs[1], order=index_order)
    axs[1].legend(loc='best')

    # average default rate per bins
    axs[2].set_title('Conversion rate per class', fontsize=11, loc='left')
    sns.barplot(data=df, y=var, x='has_subscribed', errorbar=None, ax=axs[2], order=index_order)

    for ax in axs.flat:
        ax.set(xlabel='', ylabel='')
    fig.tight_layout()
    
    
def time_class(hour):
    if hour in range(0,7):
        return 'night'
    elif hour in range(7,12):
        return 'morning'
    elif hour in range(12,15):
        return 'luncthime'
    elif hour in range(15,20):
        return 'afternoon'
    elif hour in range(20,24):
        return 'evening'
    
def delay_class(day):
    if pd.isnull(day):
        return '00'
    if day in range(0,3):
        return '0'+ str(int(day))
    elif day in range(3,8):
        return '03-07'
    elif day in range(8,15):
        return '08-14'
    elif day in range(15,29):
        return '15-28'
    else:
        return '29+'
    
      
def class_adjustment(df, region_mapping):
    
    # rbs result
    df.loc[df['rbs_result'].isin(['accepted','needs device']),'rbs_result']='accepted'
    
    # driver age
    df.loc[df['main_driver_age'].isin(['40-59','60+']),'main_driver_age']='40+'
    
    # driver licence age
    df.loc[df['main_driver_licence_age'].isin(['00','01']),'main_driver_licence_age']='00-01'
    df.loc[df['main_driver_licence_age'].isin(['03-06','07-14']),'main_driver_licence_age']='03-14'
    
    # driver bonus
    df.loc[df['main_driver_bonus'].isin(['051-063','064-084']),'main_driver_bonus']='051-084'
    
    # vehicle age
    df.loc[df['vehicle_age'].isin(['00','01-02']),'vehicle_age']='00-02'
    df.loc[df['vehicle_age'].isin(['10-14','15-19']),'vehicle_age']='10-19'
    
    # vehicle class
    df.loc[df['vehicle_class'].isin(['S-T','U+']),'vehicle_class']='S+'
    
    # vehicle group
    df.loc[df['vehicle_group'].isin(['33-34','35+']),'vehicle_group']='33+'
    df.loc[df['vehicle_group'].isin(['20-26','27-28']),'vehicle_group']='20-28'
    
    # regroupement géographique
    df['vehicle_region_aggregate'] = df['vehicle_region'].replace(region_mapping)
    
    return df


region_agg = {
'Alsace':'Est',
'Lorraine':'Est',
'Bourgogne':'Bourgogne/Franche-Comte',
'Franche-Comte':'Bourgogne/Franche-Comte',
'Haute-Normandie':'Normandie-Bretagne',
'Basse-Normandie':'Normandie-Bretagne', 
'Bretagne':'Normandie-Bretagne', 
'Pays De La Loire':'Centre-Ouest',
'Poitou-Charentes':'Centre-Ouest',
'Centre':'Centre-Ouest',
'Limousin':'Centre-Ouest',
'Auvergne':'Centre-Ouest',
'Midi-Pyrenees': 'Pyrenees - Languedoc',
'Languedoc-Roussillon': 'Pyrenees - Languedoc',
'Champagne-Ardenne':'Est',
'Picardie':'Nord',
'Nord-Pas-De-Calais':'Nord',
}



def make_shap_waterfall_plot(shap_values, features, num_display=20):
    
    '''
    A function for building a SHAP waterfall plot.
    
    SHAP waterfall plot is used to visualize the most important features in a descending order.
    
    Parameters:
    shap_values (list): SHAP values obtained from a model
    features (pandas DataFrame): a list of features used in a model
    num_display(int): number of features to display
    
    Returns:
    matplotlib.pyplot plot: SHAP waterfall plot
    
    '''
    
    column_list = features.columns
    feature_ratio = (np.abs(shap_values).sum(0) / np.abs(shap_values).sum()) * 100
    column_list = column_list[np.argsort(feature_ratio)[::-1]]
    feature_ratio_order = np.sort(feature_ratio)[::-1]
    cum_sum = np.cumsum(feature_ratio_order)
    column_list = column_list[:num_display]
    feature_ratio_order = feature_ratio_order[:num_display]
    cum_sum = cum_sum[:num_display]
    
    num_height = 0
    if (num_display >= 20) & (len(column_list) >= 20):
        num_height = (len(column_list) - 20) * 0.4
        
    fig, ax1 = plt.subplots(figsize=(8, 8 + num_height))
    ax1.plot(cum_sum[::-1], column_list[::-1], c='blue', marker='o')
    ax2 = ax1.twiny()
    ax2.barh(column_list[::-1], feature_ratio_order[::-1], alpha=0.6)
    
    ax1.grid(True)
    ax2.grid(False)
    ax1.set_xticks(np.arange(0, round(cum_sum.max(), -1)+1, 10))
    ax2.set_xticks(np.arange(0, round(feature_ratio_order.max(), -1)+1, 10))
    ax1.set_xlabel('Cumulative Ratio')
    ax2.set_xlabel('Composition Ratio')
    ax1.tick_params(axis="y", labelsize=13)
    plt.ylim(-1, len(column_list))
    

def run_model(model , X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> float:
    """ 
    fit and test estimator on a given set of train and test data.
    """
    
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_train)[:, 1]
    print('Train AUC: {}'.format(roc_auc_score(y_train, pred)))
    pred = model.predict_proba(X_test)[:, 1]
    print('Test AUC: {}'.format(roc_auc_score(y_test, pred)))

    return roc_auc_score(y_test, pred)