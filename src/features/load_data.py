import sys
import pandas as pd
import numpy as np
import json
sys.path.append('./../src/')
from etl.postgres_utils import get_select, execute_query
from project_settings import PREFIX, FEATURE_GROUPS, TEST_ID
pd.options.mode.chained_assignment = None  # default='warn'

def get_data(dataset, metric = None, feature_groups = list(FEATURE_GROUPS.keys())):
    data = None
    if metric == 'labels':
        data = get_feature_labels(dataset, feature_groups)
    return data


def get_feature_labels(dataset, feature_groups):
    part2 = "select  *"
    for feature in feature_groups:
        new_feat =  """ , {prefix}_{dataset}_%s as %s
        """ % (feature, feature)
        part2 = part2 + new_feat

    part3 = """from cohort.{dataset} """

    for feature in feature_groups:
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
    df.rename(columns = {'grade':'label-grade',
              'pass_fail':'label-pass-fail',
              'dropout':'label-dropout'},inplace=True)
    return df


def get_dataset_name(dataset_alias):
    if dataset_alias=='la18':
        dataset = 'epfl_algebrelineaire_2018'
    elif dataset_alias == 'la19':
        dataset = 'epfl_algebrelineaire_2019'
    elif dataset_alias == 'la_all':
        dataset = 'epfl_algebrelineaire'
    elif dataset_alias == 'fp':
        dataset = 'epfl_cs_210_2018_t3'
    elif dataset_alias == 'pc':
        dataset = 'epfl_cs_206_2019_t1'
    return  dataset


def get_text_files(week, dataset):
    etal = 'ensemble'
    dir_file = './../data/feature/{}_week-{}-{}'.format(week, etal, dataset)
    feat_values_dir = '{}/{}'.format(dir_file, 'feature_values.npz')
    data = np.load(feat_values_dir)
    lst = data.files

    sett_dir = '{}/{}'.format(dir_file, 'settings.txt')
    feat_settings = json.load(open(sett_dir, 'rb'))
    feature_names = feat_settings['feature_names']

    labels_dir = './../data/labels/{}.csv'.format(dataset)
    feat_labels = pd.read_csv(labels_dir)

    #Format data
    feat = data['feature_values']
    feat = np.nan_to_num(feat, nan = 0)
    return feat, feature_names, feat_labels

def get_within_period(feat, dataset_alias, within):
    if within == 'course':
        if dataset_alias == 'la_all':
            feat = feat[:,:10,:]
        elif dataset_alias == 'fp':
            feat = feat[:,:11,:]
        elif dataset_alias == 'pc':
            feat = feat[:,:14,:]
    elif within == 'exam':
        if dataset_alias == 'la_all':
            feat = feat[:,9:,:]
        elif dataset_alias == 'fp':
            feat = feat[:,10:15,:]
        elif dataset_alias == 'pc':
            feat = feat[:,13:,:]
    else:
        feat = feat[:,:15,:]
    return feat


def get_last_course(feat, dataset_alias):
    if dataset_alias == 'la_all':
        feat = feat[:,:,8]
    elif dataset_alias == 'fp':
        feat = feat[:,:,9]
    elif dataset_alias == 'pc':
        feat = feat[:,:,12]

    return feat

def normalize_data(feat):
    norms = np.linalg.norm(feat, axis=1, keepdims = True) # 214 students, 440 features
    norms[norms==0] = 1
    data_normalized  = feat.copy() / norms
    return data_normalized



def load_files(dataset_alias, group, metric='dtw',
                thresh = 60, within = 'course'):
    if group=='regularity':
        week = 'lq'
    else:
        week = 'eq'

    dataset = get_dataset_name(dataset_alias)
    feat, feature_names, feat_labels = get_text_files(week, dataset)

    if thresh:
        student_ids = filter_cohort(dataset_alias, thresh = thresh)
        print('New number of students:', len(student_ids))
        feat = np.take(feat, student_ids, axis = 0)
        feat_labels = feat_labels[feat_labels['user_index'].isin(student_ids)]

    feat = get_within_period(feat, dataset_alias, within)

    if group == 'consistency':
        feat = normalize_data(feat)

    feat = np.swapaxes(feat,0,2)
    feat = np.swapaxes(feat,1,2)

    if metric == 'euclidean':
        feat = get_last_course(feat, dataset_alias)

    print("features, students, weeks:", feat.shape)
    feat_labels['label_pass_fail'] = 0
    feat_labels['label_pass_fail'][feat_labels['label_grade'] < 4] = 1
    # feat (2, 214, 14) features, students, weeks
    return feat, feature_names, feat_labels



def get_features(feat, feature_names,features, group, metric = 'dtw'):

    features_index = [x for x in range(len(feature_names)) if feature_names[x] in features]

    if len(features)==1 and len(features_index)>1: # In case of duplicated names
        features_index = features_index[0]

    if metric == 'euclidean':
        data = feat[features_index,:]
    else:
        data = feat[features_index,:,:]
    return data.squeeze()



def filter_cohort(dataset, thresh):
    query = """
    drop table if exists cohort.{dataset};
    create table if not exists cohort.{dataset} as (
      with number_of_events as(
        select
        student
        from semantic.students_events_{dataset}
        where event_type::text like 'video.%'
        group by student
        having count(*) >= {thresh}
      )
      select * from
      semantic.students_{dataset}
      where accountuserhash in
      (select student from number_of_events)
    );
    """.format(dataset = dataset, thresh=thresh)
    execute_query(query)

    query = """
    select student from cohort.{dataset}
    order by student;
    """.format(dataset = dataset)
    df = get_select(query)
    return df.student.tolist()

