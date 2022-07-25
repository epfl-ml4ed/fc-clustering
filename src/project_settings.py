# gridsearch of features
PREFIX = 'final'
TEST_ID = PREFIX
SEED = 123
DATASETS = ['fc_la', 'fp','pc']


FEATURE_GROUPS =  {

    'effort': {'metric': 'dtw',
    'features': [
    'ch_time_sessions_sum',
    'ex_total_clicks_video'
    ]},

    'consistency': {'metric': 'dtw',
    'features': [
    'ch_time_sessions_sum',
    'ch_time_sessions_mean',
    'ex_total_clicks_video'
    ]},

    'regularity': {'metric':'euclidean',
    'features': [
    'bo_reg_periodicity_m1', # hour of the day (FDH)
    'bo_reg_periodicity_m2', # hour of the day + day of the week (FWH)
    'bo_reg_periodicity_m3']}, # day of the week (FWD)


    'proactivity': {'metric': 'dtw',
    'features': [
    'ma_content_anti',
    'bo_delay_lecture']},

    'pacing': {'metric': 'dtw',
    'features': [
    "mu_frequency_action_relative_video_pause",
    "mu_fraction_spent_completed_video_play",
    "mu_speed_playback_mean"
    ]},

    'quality': {'metric': 'dtw',
    'features': [
    'ma_competency_strength',
    'ma_student_shape']},
    }
