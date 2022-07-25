import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
import itertools as it
import time
import sys
sys.path.append('./../src/')

from project_settings import PREFIX, TEST_ID, FEATURE_GROUPS, DATASETS
from etl.postgres_utils import  get_select
from features.load_data import get_data, load_files, get_features
from features.preprocess import  format_feature
from models.clustering import plot_metrics
from tslearn.barycenters import \
    euclidean_barycenter, \
    dtw_barycenter_averaging, \
    dtw_barycenter_averaging_subgradient, \
    softdtw_barycenter


def plot_all_features(bygroup = True, feature_groups = FEATURE_GROUPS,
            overlapped_p = False, mean_p =False, median_p = False,
            separate_p = False, plot_bary_p = False, plot_heat_p = False, datasets = DATASETS):

    for dataset in datasets:
        for group in feature_groups.keys():
            metric = feature_groups[group]['metric']
            features = feature_groups[group]['features']

            feat, feature_names, feat_labels = load_files(dataset, group, metric)

            try:
                for feature in features:
                    data = get_features(feat,feature_names, [feature], group, metric)
                    data = data.squeeze()
                    feature = feature.lower().replace('-','_').replace('.','_')

                    visualize_clusters(data,  metric, feature, dataset, bygroup = bygroup,
                                        group = group,
                                        overlapped_p = overlapped_p, mean_p =mean_p,
                                        median_p = median_p, separate_p =separate_p,
                                        plot_bary_p = plot_bary_p, plot_heat_p = plot_heat_p)

                    time.sleep(5)

            except  Exception as e:
                print(e)
                print("error")


def visualize_clusters(data,  metric, feature, dataset, bygroup,
 overlapped_p, mean_p, separate_p, median_p, plot_bary_p, plot_heat_p,  group = ''):

    if plot_heat_p and metric =='dtw':
        plot_heatmap(data, feature, dataset, group)
    elif plot_heat_p and metric !='dtw':
        plot_single_boxplot(data, feature, dataset, group)

    else:
        if bygroup:
            query = """select {0}_{1}_{2} as label, *
                    from labels.{0}_{1}_{2}""".format(PREFIX,  dataset, group).lower()

        else:
            query = """select {0}_{1}_{2}_{3} as label, *
                    from labels.{0}_{1}_{2}_{3}""".format(PREFIX,  dataset, group, feature).lower()

        df_labels = get_select(query)
        labels = df_labels.label

        if metric == 'dtw':

            if overlapped_p:
                plot_overlapped(data, labels,  feature,
                 dataset, group, bygroup )

            if median_p:
                plot_median(data, labels,  feature,
                dataset, group, bygroup)

            if mean_p:
                plot_mean(data, labels,  feature,
                dataset, group, bygroup)

            if separate_p:
                plot_separate(data, df_labels.student, labels,
                 feature, dataset,group, bygroup )

            if plot_bary_p:
                plot_bary(data, labels,  feature, dataset, group, bygroup)

        else:
            df_labels[feature] = data
            plot_boxplot( df_labels,  feature, dataset, group, bygroup )


def plot_single_boxplot(data, feature, dataset, group):
    plot_dir = "./../results/features/{}/{}".format(dataset, group)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    ax = plt.axes()
    sns.boxplot(x = data, ax = ax)
    ax.set_title(feature)

    img_dir = "{0}/{1}.png".format(plot_dir, feature)
    plt.savefig(img_dir)
    plt.close()


def plot_heatmap(data, feature, dataset, group):
    plot_dir = "./../results/features/{}/{}".format(dataset, group)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    ax = plt.axes()
    sns.heatmap(data, ax = ax)
    ax.set_title(feature)

    img_dir = "{0}/{1}.png".format(plot_dir, feature)
    plt.savefig(img_dir)
    plt.close()



def plot_boxplot(df, feature, dataset,group, bygroup):

    #save image
    if bygroup:
        plot_dir = """./../results/{prefix}/{dataset}/{group}/boxplot""".format(prefix = TEST_ID, dataset= dataset, group = group)
    else:
        plot_dir = """./../results/{prefix}/{dataset}/boxplot""".format(prefix = TEST_ID, dataset= dataset, group = group)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    sns.boxplot(data = df, y = feature, x = 'label')

    img_dir = "{0}/{1}.png".format(plot_dir, feature)
    plt.savefig(img_dir)
    plt.close()

def plot_barplot(df, feature, dataset,group, bygroup):

    #save image
    if bygroup:
        plot_dir = """./../results/{prefix}/{dataset}/{group}/boxplot""".format(prefix = TEST_ID, dataset= dataset, group = group)
    else:
        plot_dir = """./../results/{prefix}/{dataset}/boxplot""".format(prefix = TEST_ID, dataset= dataset, group = group)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    #df.groupby('label', as_index=false)[feature].mean()
    sns.barplot(data = df, y = feature, x = 'label', ci = False)

    img_dir = "{0}/{1}.png".format(plot_dir, feature)
    plt.savefig(img_dir)
    plt.close()


def plot_overlapped(data, labels,  feature, dataset,group,bygroup):
    _, biweeks = data.shape
    clusters = np.unique(labels).shape[0]
    fig, axs = plt.subplots(1, clusters, figsize=(16, 4),
                            facecolor='w', edgecolor='k')
    axs = axs.ravel()
    upper = np.percentile(data, 99.5)
    upper = upper if upper > 0 else 1

    #save image
    if bygroup:
        plot_dir = """./../results/{prefix}/{dataset}/{group}/overlapped
        """.format(prefix = TEST_ID,  group = group, dataset = dataset)
    else:
        plot_dir = """./../results/{prefix}/{dataset}/overlapped
        """.format(prefix = TEST_ID,dataset =dataset )
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    dif_labels =  np.unique(labels)
    for i in range(clusters):
        students_cluster = data[labels == dif_labels[i]]
        number_students = students_cluster.shape[0]

        for student in range(number_students):
            axs[i].bar(range(biweeks), students_cluster[student], alpha=0.3)
        axs[i].set_ylim([0, upper])
        axs[i].set_title('Group {0}, students: {1}'.format(i, number_students))
        axs[i].set_ylabel('{}_{}'.format(feature, dataset))
        axs[i].set_xlabel('Week of the course')

    img_dir = "{0}/{1}_{2}.png".format(plot_dir, feature,  dataset)
    plt.savefig(img_dir)
    plt.close()



def plot_median(data, labels,  feature, dataset, group, bygroup,
            gridsearch = False, gamma = 1, window = 1):
    _, biweeks = data.shape
    clusters = np.unique(labels).shape[0]
    fig, axs = plt.subplots(1, clusters, figsize=(16, 4),
                            facecolor='w', edgecolor='k')
    axs = axs.ravel()

    #save image
    if bygroup:
        plot_dir = """./../results/{prefix}/{dataset}/{group}/median
        """.format(prefix =TEST_ID,  group = group, dataset = dataset)
    elif gridsearch:
        plot_dir = """./../results/{prefix}/{dataset}/median/gridsearch/{feature}/{clusters}
        """.format(prefix = TEST_ID,  feature = feature, clusters = clusters,  dataset = dataset)
    else:
        plot_dir = """./../results/{prefix}/{dataset}/median""".format(prefix = TEST_ID, dataset=dataset)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    upper = np.percentile(data, 95)
    upper = upper if upper > 0 else 1

    lower = np.percentile(data, 1)
    lower = -1 if lower < 0 else 0

    dif_labels =  np.unique(labels)

    for i in range(clusters):
        students_cluster = data[labels == dif_labels[i]]
        number_students = students_cluster.shape[0]
        values =  np.median(students_cluster, axis = 0)

        axs[i].bar(range(biweeks), values, alpha=0.3)
        axs[i].set_ylim([lower, upper])
        axs[i].set_title('Group {0}, students: {1}, mean: {2}'.format(i, number_students, round(np.mean(values),3)))
        axs[i].set_ylabel(feature)
        axs[i].set_xlabel('Week of the course')

    if gridsearch:
        img_dir = "{0}/{1}_{2}_G{3}_W{4}.png".format(plot_dir, feature,  dataset, gamma, window)
    else:
        img_dir = "{0}/{1}_{2}.png".format(plot_dir, feature,  dataset)
    plt.savefig(img_dir)
    plt.close()


def plot_helper(barycenter):
    # plot all points of the data set
    for series in X:
        plt.plot(series.ravel(), "k-", alpha=.2)
    # plot the given barycenter of them
    plt.plot(barycenter.ravel(), "r-", linewidth=2)


def plot_bary(data, labels,  feature, dataset, group, bygroup):
    _, biweeks = data.shape
    clusters = np.unique(labels).shape[0]
    fig, axs = plt.subplots(1, clusters, figsize=(16, 4),
                            facecolor='w', edgecolor='k')
    axs = axs.ravel()

    #save image
    if bygroup:
        plot_dir = """./../results/{prefix}/{dataset}/{group}/bary
        """.format(prefix =TEST_ID,  group = group, dataset = dataset)
    else:
        plot_dir = """./../results/{prefix}/{dataset}/bary""".format(prefix = TEST_ID, dataset = dataset )
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    upper = np.percentile(data, 96)
    upper = upper if upper > 0 else 1

    lower = np.percentile(data, 1)
    lower = -1 if lower < 0 else 0

    dif_labels =  np.unique(labels)

    for i in range(clusters):
        students_cluster = data[labels == dif_labels[i]]
        number_students = students_cluster.shape[0]

        #for student in range(number_students):
        #    axs[i].plot(range(biweeks), students_cluster[student], alpha=0.15)
        values =  np.median(students_cluster, axis = 0)
        axs[i].bar(range(biweeks), values, alpha=0.3)

        barycenter = softdtw_barycenter(students_cluster, gamma=1, max_iter=100, tol=1e-3)
        axs[i].plot(barycenter.ravel(), "r-", linewidth=2)
        axs[i].set_ylim([0, upper])

        axs[i].set_title('Group {0}, students: {1}'.format(i, number_students))
        axs[i].set_ylabel('{}_{}'.format(feature, dataset))
        axs[i].set_xlabel('Week of the course')


    img_dir = "{0}/{1}_{2}.png".format(plot_dir, feature,  dataset)
    plt.savefig(img_dir)
    plt.close()


def plot_mean(data, labels,  feature, dataset, group, bygroup,
            gridsearch = False, gamma = 1, window = 1 ):
    _, biweeks = data.shape
    clusters = np.unique(labels).shape[0]
    fig, axs = plt.subplots(1, clusters, figsize=(16, 4),
                            facecolor='w', edgecolor='k')
    axs = axs.ravel()

    #save image
    if bygroup:
        plot_dir = """./../results/{prefix}/{dataset}/{group}/average
        """.format(prefix =TEST_ID,  group = group, dataset = dataset)
    elif gridsearch:
        plot_dir = """./../results/{prefix}/{dataset}/average/gridsearch/{feature}/{clusters}
        """.format(prefix = TEST_ID,  feature = feature, clusters = clusters,dataset = dataset )
    else:
        plot_dir = """./../results/{prefix}/{dataset}/average""".format(prefix = TEST_ID, dataset = dataset )
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    upper = np.percentile(data, 96)
    upper = upper if upper > 0 else 1

    lower = np.percentile(data, 1)
    lower = -1 if lower < 0 else 0

    dif_labels =  np.unique(labels)

    for i in range(clusters):
        students_cluster = data[labels == dif_labels[i]]
        number_students = students_cluster.shape[0]
        values =  np.mean(students_cluster, axis = 0)

        axs[i].bar(range(biweeks), values, alpha=0.3)
        axs[i].set_ylim([lower, upper])
        axs[i].set_title('Group {0}, students: {1}, mean: {2}'.format(dif_labels[i], number_students, round(np.mean(values),3)))
        axs[i].set_ylabel(feature)
        axs[i].set_xlabel('Week of the course')

    if gridsearch:
        img_dir = "{0}/{1}_{2}_G{3}_W{4}.png".format(plot_dir, feature,  dataset, gamma, window)
    else:
        img_dir = "{0}/{1}_{2}.png".format(plot_dir, feature,  dataset)
    plt.savefig(img_dir)
    plt.close()



def plot_separate(data, students_ids, labels,  feature, dataset,group, bygroup):
    _, biweeks = data.shape
    clusters = np.unique(labels).shape[0]
    upper = np.percentile(data, 99.5)
    upper = upper if upper > 0 else 1

    lower = np.percentile(data, 1)
    lower = -1 if lower < 0 else 0
    dif_labels =  np.unique(labels)

    for i in range(clusters):
        students_cluster = data[labels == dif_labels[i]]
        student_ids_list = students_ids[labels == i].values
        number_students = students_cluster.shape[0]

        #save image
        if bygroup:
            plot_dir = """./../results/{prefix}/{dataset}/{group}/individual/{feature}/cluster_{i}
            """ .format(prefix =TEST_ID,  feature= feature, dataset =dataset, i = i,  group = group,)
        else:
            plot_dir = """./../results/{prefix}/{dataset}/individual/{feature}/cluster_{i}
            """ .format(prefix = TEST_ID,  feature= feature, dataset =dataset, i = i, )
        Path(plot_dir).mkdir(parents=True, exist_ok=True)

        for student in range(number_students):
            plt.bar(range(biweeks), students_cluster[student], alpha=0.3)
            plt.ylim([lower, upper])
            plt.title('Group {0}, student: {1}'.format(i,student_ids_list[student]))
            plt.ylabel(feature)

            img_dir = "{0}/{1}.png".format(plot_dir, student_ids_list[student])
            plt.savefig(img_dir)
            plt.close()
            time.sleep(2)





