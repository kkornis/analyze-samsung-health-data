import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


'''The data is updated at 20230722102693'''
as_of_date = '20230722102693'


def get_file_name(short, s):
    direct = os.path.join(os.path.dirname(__file__), 'data')
    return os.path.join(direct, 'com.samsung.' + ('s' if s else '') + 'health.' + short + '.csv')


def mv_files():
    direct = os.path.join(os.path.dirname(__file__), 'data')
    files = os.listdir(direct)
    files = [f for f in files if f.endswith('.csv')]

    for file in files:
        joint_path = os.path.join('data', file)
        joint_new_path = joint_path[:-19] + '.csv'
        os.system('git -C K:/Kristof/prg/analyze_heart_rate_data mv ' + joint_path + ' ' + joint_new_path)


def rename_files():
    direct = os.path.join(os.path.dirname(__file__), 'data')
    files = os.listdir(direct)
    files = [f for f in files if f.endswith('.csv')]

    for file in files:
        joint_path = os.path.join(direct, file)
        new_file = file[:-19] + '.csv'
        joint_new_path = os.path.join(direct, new_file)
        os.rename(joint_path, joint_new_path)


def my_lambda(x):
    if x.startswith('com.samsung.health.sleep.'):
        x = 's.s.' + x[25:]
    if x.startswith('com.samsung.health.exercise.'):
        x = 's.e.' + x[28:]
    if x.startswith('com.samsung.health.heart_rate.'):
        x = 's.h.' + x[30:]
    return x


def get_data(short):
    fname = get_file_name(short, s=True)
    file = open(fname, "r")
    txt = file.read()
    # txt.replace('\n,', '\n')
    lines = txt.split('\n')

    df = pd.DataFrame([line.split(',')[:-1] for line in lines[2:-1]], columns=lines[1].split(','))
    df.rename(my_lambda, inplace=True, axis='columns')
    return df


def get_sleep_data_table():
    data_name = 'sleep'
    df = get_data(data_name)
    cols = df.columns
    dates_cols = ['s.s.start_time', 's.s.end_time']
    meaningful_cols = ['mental_recovery', 'physical_recovery', 'sleep_score', 'movement_awakening', 'sleep_cycle', 'efficiency', 'sleep_duration']
    semi_meaningful_cols = ['factor_' + str(i).zfill(2) for i in range(1, 11)]
    meaningless_col = ['has_sleep_data', 'data_version']
    empty_cols = ['sleep_type', 'original_wake_up_time', 'original_bed_time', 'original_efficiency', 'quality']
    id_cols = ['combined_id', 'extra_data']
    col_types = [dates_cols, meaningful_cols, semi_meaningful_cols, meaningless_col, empty_cols, id_cols]
    remaining = [rem for rem in cols if not any([rem in x for x in col_types])]
    col_types = col_types + [remaining]
    cols = sum(col_types, [])
    df = df[cols]
    for col in meaningful_cols + semi_meaningful_cols + meaningless_col:
        df[col] = pd.to_numeric(df[col])
    for col in dates_cols:
        df[col] = pd.to_datetime(df[col])
    df.sort_values(by='s.s.start_time', inplace=True)
    return df


def play_with_sleep_data():
    df0 = get_sleep_data_table()

    df2 = get_data('sleep_combined')
    df3 = get_data('sleep_goal')

    df0['sleep_duration_in_h'] = df0['sleep_duration'] / 60
    df0['s.s.start_time_date'] = df0['s.s.start_time'].dt.date

    # when I should never sleep
    midday = 14
    time_zone_delay = 2
    ser_h = df0['s.s.start_time'].dt.hour + time_zone_delay - midday
    ser_h = np.where(ser_h < 0, ser_h + 24, ser_h) + midday - 24
    df0['s.s.start_time_time'] = ser_h + df0['s.s.start_time'].dt.minute / 60

    fig, (ax1, ax2) = plt.subplots(2, 1)
    df0.plot(kind='bar', x='s.s.start_time_date', y='sleep_duration_in_h', bottom=df0['s.s.start_time_time'], ax=ax1)
    ax1.hlines(y=[-1, 7], xmin=0, xmax=len(df0), colors=['r', 'r'])

    df0.plot(kind='bar', x='s.s.start_time_date', y='sleep_score', ax=ax2, color='g')

    # plt.show()
    return df0


def get_exercise_table():
    data_name = 'exercise'
    df = get_data(data_name)
    cols = df.columns
    dates_cols = ['s.e.start_time', 's.e.end_time']
    meaningful_cols = ['total_calorie', 'heart_rate_sample_count', 's.e.duration', 's.e.exercise_type',
                       's.e.min_altitude', 's.e.max_altitude', 's.e.mean_heart_rate', 's.e.count', 's.e.distance',
                       's.e.calorie', 's.e.mean_speed', 's.e.altitude_gain', 's.e.sweat_loss', 's.e.min_heart_rate',
                       's.e.max_heart_rate', 's.e.max_speed', 's.e.mean_cadence', 's.e.max_cadence',
                       's.e.decline_distance', 's.e.vo2_max', 's.e.incline_distance', 'source_type', 'reward_status',
                       's.e.count_type']
    semi_meaningful_cols = ['s.e.comment']
    meaningless_col = []
    empty_cols = ['mission_value', 'subset_data', 'routine_datauuid', 'pace_info_id', 'pace_live_data']
    id_cols = ['live_data_internal', 'sensing_status', 'location_data_internal', 'additional_internal',
               's.e.location_data', 's.e.live_data']
    col_types = [dates_cols, meaningful_cols, semi_meaningful_cols, meaningless_col, empty_cols, id_cols]
    remaining = [rem for rem in cols if not any([rem in x for x in col_types])]
    col_types = col_types + [remaining]
    cols = sum(col_types, [])
    df = df[cols]
    for col in meaningful_cols + meaningless_col:
        df[col] = pd.to_numeric(df[col])
    for col in dates_cols:
        df[col] = pd.to_datetime(df[col])
    df.sort_values(by='s.e.start_time', inplace=True)
    return df


def play_with_exercise_data():
    # mean_speed is in m/s
    df = get_exercise_table()
    df_walk = df[df['s.e.exercise_type'] == 1001]
    df_run = df[df['s.e.exercise_type'] == 1002]
    df_pull_ups = df[df['s.e.exercise_type'] == 10005]
    df_cycling = df[df['s.e.exercise_type'] == 11007]
    df_hike = df[df['s.e.exercise_type'] == 13001]
    df_swim_outdoor = df[df['s.e.exercise_type'] == 14001]
    15002

    df_run['s.e.max_speed_kmh'] = df_run['s.e.max_speed'] * 3.6

    df_run['s.e.start_time_date'] = df_run['s.e.start_time'].dt.date
    df_run.plot(kind='bar', x='s.e.start_time_date', y='s.e.max_speed_kmh')

    return df


def get_heart_rate_table():
    data_name = 'tracker.heart_rate'
    df = get_data(data_name)
    cols = df.columns
    dates_cols = ['s.h.start_time', 's.h.end_time']
    meaningful_cols = ['s.h.min', 's.h.max', 's.h.heart_rate', 's.h.heart_beat_count']
    semi_meaningful_cols = ['s.h.comment']
    meaningless_col = []
    empty_cols = ['source', 's.h.custom']
    id_cols = ['tag_id', 's.h.binning_data', 's.h.update_time', 's.h.create_time',
               's.h.time_offset', 's.h.deviceuuid']
    col_types = [dates_cols, meaningful_cols, semi_meaningful_cols, meaningless_col, empty_cols, id_cols]
    remaining = [rem for rem in cols if not any([rem in x for x in col_types])]
    col_types = col_types + [remaining]
    cols = sum(col_types, [])
    df = df[cols]
    for col in meaningful_cols + meaningless_col:
        df[col] = pd.to_numeric(df[col])
    for col in dates_cols:
        df[col] = pd.to_datetime(df[col])
    df.sort_values(by='s.h.start_time', inplace=True)
    return df


def play_with_heart_rate_data():
    df = get_heart_rate_table()
    df['diff'] = df['s.h.max'] - df['s.h.min']
    df['s.h.start_time_date'] = df['s.h.start_time'].dt.date

    fig, ax = plt.subplots(1, 1)
    # df.plot(kind='bar', x='s.h.start_time_date', y='diff', bottom=df['s.h.min'])
    df.plot(x='s.h.start_time', y=['s.h.heart_rate', 's.h.min', 's.h.max'], ax=ax)
    ax.hlines(y=50, xmin=df['s.h.start_time'].iloc[0], xmax=df['s.h.start_time'].iloc[len(df)-1], colors='r')
    plt.show()
    return df


def main():
    play_with_sleep_data()
    play_with_exercise_data()
    play_with_heart_rate_data()


if __name__ == '__main__':
    main()

