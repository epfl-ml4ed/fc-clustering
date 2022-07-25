import pandas as pd
import numpy as np
import itertools as it
from scipy.stats import ranksums
from scipy import stats
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency

import sys
sys.path.append('./../src/')
from models.clustering import compute_bic
from etl.postgres_utils import get_select
from project_settings import FEATURE_GROUPS, TEST_ID
from features.load_data import get_data
pd.options.mode.chained_assignment = None  # default='warn'

def get_scores(kmeans, kernel_matrix, proj_X, distances, metadata):
    y_pred = kmeans.labels_

    metadata['s_distortion'] = kmeans.inertia_
    metadata['s_bic'] = compute_bic(kmeans, proj_X)

    if len(np.unique(y_pred)) == 1:
        print("Clustering error. Can't compute Silhouette correctly")
        metadata['s_sil_km'] = 0
        metadata['s_sil_dist'] = 0
    else:
        metadata['s_sil_km'] = silhouette_score(proj_X, y_pred)
        metadata['s_sil_dist'] = silhouette_score(distances, y_pred, metric = 'precomputed')

    return metadata



def get_group_composition_np(df, clusters, metadata):
    grade_list = []
    pass_list = []
    dropout_list = []
    gender_list = []
    cat_list = []
    round_list = []

    df['label_pass_fail'] = 0
    df['label_pass_fail'][df['label_grade'] > 4] = 1

    obs = pd.crosstab(df['label'], df['label_pass_fail'], margins = False)
    chi2, p, dof, ex = chi2_contingency(obs, correction=False)
    metadata['chi2_pass'] = p

    obs = pd.crosstab(df['label'], df['label_dropout'], margins = False)
    chi2, p, dof, ex = chi2_contingency(obs, correction=False)
    metadata['chi2_dropout'] = p

    df = df.rename({'round_id':'round'}, axis = 1)

    if metadata['dataset'] in ('la_all', 'la18', 'la19'):
        obs = pd.crosstab(df['label'], df['gender'], margins = False)
        chi2, p, dof, ex = chi2_contingency(obs, correction=False)
        metadata['chi2_gender'] = p

        obs = pd.crosstab(df['label'], df['round'], margins = False)
        chi2, p, dof, ex = chi2_contingency(obs, correction=False)
        metadata['chi2_round'] = p

        obs = pd.crosstab(df['label'], df['category'], margins = False)
        chi2, p, dof, ex = chi2_contingency(obs, correction=False)
        metadata['chi2_category'] = p

    clusters_list = np.unique(df['label'])
    for c in clusters_list:
        category = df[df['label']==c]
        rest = df[df['label']!=c]

        feature_category = category['label_grade'].dropna()
        feature_rest = rest['label_grade'].dropna()
        _, pvalue = ranksums(feature_category, feature_rest)
        grade_list.append(pvalue)

        feature = category['label_pass_fail'].dropna()
        percentage = np.mean(feature)
        pass_list.append(percentage)

        feature = category['label_dropout'].dropna()
        percentage = np.mean(feature)
        dropout_list.append(percentage)

        if metadata['dataset'] in ('la_all', 'la18', 'la19'):
            feature = category['gender'].dropna()
            percentage = np.divide(sum(feature == 'F'),len(feature))
            gender_list.append(percentage)

            feature = category['round'].dropna()
            percentage = np.divide(sum(feature == 'Y2-2018-19'),len(feature))
            round_list.append(percentage)

            feature = category['category'].dropna()
            percentage = np.divide(sum(feature == 'France'),len(feature))
            cat_list.append(percentage)


    metadata['grade_list'] = grade_list #sorted(grade_list)
    metadata['pass_ratio_list'] = pass_list #sorted(pass_list)
    metadata['dropout_list'] = dropout_list # sorted(dropout_list)

    metadata['gender_list'] = gender_list # sorted(gender_list)
    metadata['round_list'] =  round_list #sorted(round_list)
    metadata['category_list'] = cat_list#sorted(cat_list)

    return metadata


def labels_correlation(df, feature_groups = list(FEATURE_GROUPS.keys())):
    """
    Adapted from: https://medium.com/analytics-vidhya/constructing-heat-map-for-chi-square-test-of-independence-6d78aa2b140f

    If the expected frequency is less than 5 for the (20%) of the group of frequencies between two variables we will ignore the p-value between those two variables while inspecting the heat map visually. Here we are not ignoring variables rather we don’t trust the p-value between them due to the low count of frequency so ideally, we will keep both variables. I am assigning relatively a different numeric value over here to such variables that value is “2”.

    Since the p-value between “gender” and “relation” is less than our chosen significance level of (α = 0.05), we can reject the null hypothesis. We can conclude that there is enough evidence to suggest an association between gender and relation.
    Since the p-value between “secionId” and “class” is greater than our chosen significance level of (α = 0.05), we can not reject the null hypothesis. We can conclude that there is NOT enough evidence to suggest an association between gender and relation.
    """

    # Assiging column names to row indexs
    chisqmatrix=pd.DataFrame(df,columns=feature_groups,index=feature_groups)
    num_groups = len(feature_groups)

    for i in range(num_groups): # Outer loop
        for j in range(i+1,num_groups): # inner loop
            icol = feature_groups[i]
            jcol = feature_groups[j]
            # Converting to cross tab as for CHi-square test we have
            # to first convert variables into contigency table
            mycrosstab=pd.crosstab(df[icol],df[jcol])
            #Getting p-value and other usefull information
            stat,p,dof,expected=stats.chi2_contingency(mycrosstab)
            # Rounding very small p-values to zero
            chisqmatrix.iloc[i,j]=round(p,5)
            # As mentioned above Expected frequencies should be at
            # least 5 for the majority (80%) of the cells.
            # Here we are checking expected frequency of each group
            cntexpected=expected[expected<5].size
            #Getting percentage
            perexpected=((expected.size-cntexpected)/expected.size)*100

            if perexpected<20:
                chisqmatrix.iloc[i,j]=2 #Assigning 2

    return chisqmatrix


def grades_pval_pairwise(df, criteria = 'label_grade'):
    clusters = np.unique(df['label'])
    # Assiging column names to row indexs
    pmatrix=pd.DataFrame(columns=clusters,index=clusters)
    num_groups = len(clusters)

    for i in range(num_groups): # Outer loop
        for j in range(i+1,num_groups): # inner loop

            category = df[df['label']==clusters[i]]
            rest = df[df['label']==clusters[j]]

            feature_category = category[criteria].dropna()
            feature_rest = rest[criteria].dropna()
            _, pvalue = ranksums(feature_category, feature_rest)
            pmatrix.iloc[i,j] = pvalue

    return pmatrix

def grades_pval(df, features_names):
    grades = []
    for feat in features_names:
        category = df[df[feat]==0]
        rest = df[df[feat]!=0]

        feature_category = category['label_grade'].dropna()
        feature_rest = rest['label_grade'].dropna()
        _, pvalue = ranksums(feature_category, feature_rest)
        grades.append(pvalue)

    df_grades = pd.DataFrame({'feature':features_names, 'grades':grades})
    return df_grades


def feature_group_info(dataset, feature_groups = FEATURE_GROUPS):
    features_names = list(feature_groups.keys())
    df = get_data(dataset = dataset, metric = 'labels',
                    feature_groups = features_names)
    grades = grades_pval(df, features_names)
    heat = labels_correlation(df, features_names)
    return heat, grades


def features_info(feature_groups, dataset):
    features_names = []
    for group in feature_groups.keys():
        features = feature_groups[group]['features']
        for feat in features:
            feat_name = "{}_{}".format(group, feat.lower().replace('.','_').replace('-','_'))
            features_names.append(feat_name)


    part2 = "select  *"
    part3 = """from cohort.{dataset} """

    for feature in features_names:
        new_feat =  """ , {prefix}_{dataset}_%s as %s
        """ % (feature, feature)
        part2 = part2 + new_feat

        new_feat =  """
        left join labels.{prefix}_{dataset}_%s
            using(student)
        """ % (feature)
        part3 = part3 + new_feat

    part4 = """
        order by student
    """

    query = part2 + part3 + part4
    query = query.format(prefix = TEST_ID,dataset=dataset.lower())
    df = get_select(query)
    df = df.rename({'grade':'label-grade'}, axis = 1)
    heat = labels_correlation(df, features_names)
    grades =  grades_pval(df, features_names)

    return heat, grades




def get_pairwise_wilcoxon(df, cluster_labels):
    media1_list = []
    media2_list = []
    sum_medias_list = []

    for a, b in it.combinations(np.unique(cluster_labels),2):
        category = df[cluster_labels==a]
        rest = df[cluster_labels==b]

        feature_category = category['media1'].dropna()
        feature_rest = rest['media1'].dropna()
        _, pvalue = ranksums(feature_category, feature_rest)
        media1_list.append(pvalue)

        feature_category = category['media2'].dropna()
        feature_rest = rest['media2'].dropna()
        _, pvalue = ranksums(feature_category, feature_rest)
        media2_list.append(pvalue)

        feature_category = category['sum_medias'].dropna()
        feature_rest = rest['sum_medias'].dropna()
        _, pvalue = ranksums(feature_category, feature_rest)
        sum_medias_list.append(pvalue)

    return sorted(media1_list), sorted(media2_list), sorted(sum_medias_list)
