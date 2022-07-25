import sys
sys.path.append('./../src/')
from etl.postgres_utils import execute_query, copy_df, insert_query, insert_df
from datetime import datetime
from psycopg2.extensions import register_adapter, AsIs
import numpy as np


def save_metadata(metadata, table_name ='results.gridsearch'):
    timestamp = datetime.now()
    values = (timestamp,) + tuple(metadata.values())

    num_cols = len(values)
    query = """insert into {table_name}
    values (%s {new_cols}) """.format(table_name = table_name,
                                     new_cols = ', %s'*(num_cols -1) )
    insert_query(query, values)


def create_results_table(experiment_name,features,plot=False ):
    if plot:
        dtype = 'text'
    else:
        dtype = 'int'

    vars = " "
    for feature in features[2:]:
        new_var = ", {0} int".format(feature)
        vars = vars + new_var


    query = """
        drop table if exists labels.{experiment_name};
        create table  if not exists labels.{experiment_name}(
          student int,
          {experiment_name} {dtype}
          {vars}
        );
        """.format(experiment_name = experiment_name, dtype = dtype, vars = vars)
    execute_query(query)


def save_labels(df, experiment_name, time_unit = 'biweek', plot = False):
    extra = []
    features = ['student','label'] + extra
    results = df[features]

    create_results_table(experiment_name, features, plot)
    table_name = "labels.{}".format(experiment_name)
    insert_df(results, table_name)
    print(str(table_name) + " created")
