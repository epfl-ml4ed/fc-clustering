import pandas as pd
import numpy as np
import itertools as it

from etl.postgres_utils import get_select
from sklearn.metrics import  f1_score

from project_settings import PREFIX, CLUSTER_METRIC, TEST_ID, FEATURE_GROUPS




def get_optimal_params(experiment_name, table_name = 'results.gridsearch',
                        metric = 's_sil_km desc'):



    query_generic = """
        with latest as (
            select distinct on (experiment_name, clusters, window_size, gamma) *
            from {table_name}
            where experiment_name = '{experiment_name}'
            order by experiment_name, clusters, window_size, gamma, experiment_date desc
        )
        select *
        from latest
        order by {metric}
        limit 1
        """
    query = query_generic.format(experiment_name = experiment_name,
                                 table_name = table_name,
                                 metric = metric)
    params = get_select(query)

    if len(params) == 0:
        print(query)
        print("Error: This shouldn't happen. Make sure the table is not empty.")
        print("This is the missing table", experiment_name)
        print("Set search parameter to True")
        gamma = 1
        window = 1
        exp_id = None
        optimal_clusters =  2
    else:
        gamma = params.gamma[0]
        window = params.window_size[0]
        optimal_clusters =  params.clusters[0]
        exp_id = params.experiment_date[0]
        print("Clusters: {}, Gamma: {}, Window: {}".format(optimal_clusters, gamma, window))

    return optimal_clusters, gamma, window, exp_id
