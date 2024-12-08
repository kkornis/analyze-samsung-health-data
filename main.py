"""
This module intended to provide useful parsing and some visualization od samsung health data
"""

import os
import json
import datetime as dt
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def mv_files(direct, old_timestamp: str, new_timestamp: str):
    """
    This function helps git operations of the underlying data.
    This is a deprecated function, not part of the normal workflow, no need to use it.

    :param direct: input folder path
    :param old_timestamp: timestamp to rename
    :param new_timestamp: new addition to the filename, it should be ''
    :return: None
    """

    files = os.listdir(direct)
    files = [f for f in files if f.endswith('.csv')]
    if not all(f[-18:-4] == old_timestamp for f in files):
        for f in files:
            if f[-18:-4] != old_timestamp:
                print(f, f[-18:-4], old_timestamp)

        raise ValueError

    for file in files:
        joint_new_path = file[:-19] + new_timestamp + '.csv'
        command = 'git -C ' + direct + ' mv ' + file + ' ' + joint_new_path
        print(command)
        os.system(command)


def rename_files(extra_path: str) -> None:
    """
    This function helps update the underlying data. It removes the annoying timestamps from the end of the filenames.

    :param: extra_path: additional segments of input folder.
    :return: None
    """

    direct = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'samsung_health_galaxy5_watch_data',
                                          extra_path))
    items = os.listdir(direct)
    files = [item for item in items if os.path.isfile(os.path.join(direct, item))]
    directories = [item for item in items if not os.path.isfile(os.path.join(direct, item))]
    if not all(f.endswith('.csv') for f in files):
        print('Warning, unexpected file extension!')
    if not len(directories) == 1 or directories[0] != 'jsons':
        print('Warning, unexpected directories!')

    files = [f for f in files if f.endswith('.csv')]

    for file in files:
        joint_path = os.path.join(direct, file)
        new_file = file[:-19] + '.csv'
        joint_new_path = os.path.join(direct, new_file)
        os.rename(joint_path, joint_new_path)


class HealthDataTable:
    """
    Abstract base class. Contains some basic row name rules, data directory name, etc...
    """

    data_dir = None
    csv_data_name = None

    index_col = 's.start_time'
    dates_cols = [index_col, 's.end_time']
    primary_columns = []
    secondary_numeric_columns = []
    secondary_str_columns = []
    empty_cols = []
    id_cols = []
    all_cols_ordered = []

    prefix = 's.'

    timestamp = ''

    start_time = dt.datetime(2021, 7, 23, 15)
    end_time = dt.datetime(2027, 7, 25, 5)

    def __init__(self, data_dir: str):
        HealthDataTable.data_dir = data_dir

    @staticmethod
    def manipulate_data_type(data_type):
        if not data_type.startswith('com.'):
            if data_type.startswith('s.'):
                data_type = 'shealth.' + data_type[2:]
            else:
                data_type = 'health.' + data_type
            data_type = 'com.samsung.' + data_type
        return data_type

    @staticmethod
    def get_csv_file_name(data_type):
        return os.path.join(HealthDataTable.data_dir, HealthDataTable.manipulate_data_type(data_type)
                            + HealthDataTable.timestamp + '.csv')

    @staticmethod
    def get_json_file_name(hash_data, data_type):
        starting_letter = hash_data[0]
        return os.path.join(HealthDataTable.data_dir, 'jsons', HealthDataTable.manipulate_data_type(data_type),
                            str(starting_letter), hash_data)

    @staticmethod
    def simplify_column(col_name: str) -> str:
        raise NotImplementedError

    @classmethod
    def get_data(cls) -> pd.DataFrame:
        file_name = HealthDataTable.get_csv_file_name(cls.prefix + cls.csv_data_name)
        with open(file_name, "r", encoding="utf-8") as file:
            txt = file.read()
            # txt.replace('\n,', '\n')
            lines = txt.split('\n')

            fline = lines[1] + ','
            txt_new = '\n'.join([lines[0]] + [fline] + lines[2:])

            # df = pd.DataFrame([line.split(',')[:-1] for line in lines[2:-1]], columns=lines[1].split(','))
            df = pd.read_csv(io.StringIO(txt_new), skiprows=1)
            df.rename(cls.simplify_column, inplace=True, axis='columns')
            return df

    def get_df(self) -> pd.DataFrame:
        return self.get_data()

    def get_formatted_df(self) -> pd.DataFrame:
        df = self.get_data()
        cols = df.columns
        col_types = [self.dates_cols, self.primary_columns, self.secondary_numeric_columns, self.secondary_str_columns,
                     self.empty_cols, self.id_cols]
        remaining = [rem for rem in cols if not any(rem in x for x in col_types)]
        col_types = col_types + [remaining]
        cols = sum(col_types, [])
        df = df[cols]
        for col in self.primary_columns + self.secondary_numeric_columns:
            df[col] = pd.to_numeric(df[col])
        for col in self.dates_cols:
            df[col] = pd.to_datetime(df[col])
        df.sort_values(by=self.index_col, inplace=True, ignore_index=True)
        df = df[((df[self.index_col] > HealthDataTable.start_time) &
                 (df[self.index_col] < HealthDataTable.end_time))]
        return df

    @staticmethod
    def get_json_data(hash_data, data_type):
        file_path = HealthDataTable.get_json_file_name(hash_data, data_type)
        df = pd.read_json(file_path)
        return df

    @staticmethod
    def get_complex_json_data(hash_data, data_type):
        file_path = HealthDataTable.get_json_file_name(hash_data, data_type)
        with open(file_path, 'r', encoding="utf-8") as data_file:
            data = json.load(data_file)
            in_data = data['data'][0]
            advanced_metrics = in_data['advanced_metrics']
            df_data = {'data_type': [], 'overall_score': [], 'score_ratio': []}
            keys = df_data.keys()
            dfl = []
            for line in advanced_metrics:
                for key in keys:
                    df_data[key].append(line[key])
                dfl.append(pd.DataFrame(line['chart_data']))
            df = pd.DataFrame(df_data)
            return dfl, df, in_data['sampling_rate'], in_data['service_name']


class SleepCombined(HealthDataTable):
    csv_data_name = 'sleep_combined'

    def get_formatted_df(self) -> pd.DataFrame:
        return self.get_df()

    @staticmethod
    def simplify_column(col_name):
        if col_name.startswith('com.samsung.health.sleep.'):
            col_name = 's.' + col_name[25:]
        return col_name


class SleepGoal(HealthDataTable):
    csv_data_name = 'sleep_goal'

    def get_formatted_df(self) -> pd.DataFrame:
        return self.get_df()

    @staticmethod
    def simplify_column(col_name):
        if col_name.startswith('com.samsung.health.sleep.'):
            col_name = 's.' + col_name[25:]
        return col_name

    def play_with_sleep_data(self):
        df = self.get_formatted_df()
        return df


class Sleep(HealthDataTable):
    csv_data_name = 'sleep'

    primary_columns = ['mental_recovery', 'physical_recovery', 'sleep_score', 'movement_awakening', 'sleep_cycle',
                       'efficiency', 'sleep_duration']
    secondary_numeric_columns = ['factor_' + str(i).zfill(2) for i in range(1, 11)] + ['has_sleep_data', 'data_version']
    secondary_str_columns = []
    empty_cols = ['sleep_type', 'original_wake_up_time', 'original_bed_time', 'original_efficiency', 'quality']
    id_cols = ['combined_id', 'extra_data']

    def play_with_sleep_data(self):
        df = self.get_formatted_df()
        if len(df) == 0:
            return df

        df['sleep_duration_in_h'] = df['sleep_duration'] / 60
        df['s.start_time_date'] = df['s.start_time'].dt.date

        # when I should never sleep
        midday = 14
        time_zone_delay = 2
        ser_h = df['s.start_time'].dt.hour + time_zone_delay - midday
        ser_h = np.where(ser_h < 0, ser_h + 24, ser_h) + midday - 24
        df['s.start_time_time'] = ser_h + df['s.start_time'].dt.minute / 60

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.tight_layout()
        df.plot(kind='bar', x='s.start_time_date', y='sleep_duration_in_h', bottom=df['s.start_time_time'], ax=ax1)
        ax1.hlines(y=[-1, 7], xmin=0, xmax=len(df), colors=['r', 'r'])
        ax1.hlines(y=[0, 1, 2, 3, 4, 5, 6], xmin=0, xmax=len(df), colors=['r', 'r'], linewidth=0.5)

        df.plot(kind='bar', x='s.start_time_date', y='sleep_score', ax=ax2, color='g')
        return df

    @staticmethod
    def simplify_column(col_name):
        if col_name.startswith('com.samsung.health.sleep.'):
            col_name = 's.' + col_name[25:]
        return col_name


class Stress(HealthDataTable):
    csv_data_name = 'stress'

    all_cols_ordered = ['start_time', 'custom', 'binning_data', 'tag_id', 'update_time', 'create_time', 'max', 'min',
                        'score', 'algorithm', 'time_offset', 'deviceuuid', 'comment', 'pkg_name', 'end_time',
                        'datauuid', 'Unnamed: 16']

    index_col = 'start_time'
    dates_cols = [index_col, 'end_time']

    primary_columns = ['max', 'min', 'score']
    secondary_numeric_columns = []
    secondary_str_columns = []
    empty_cols = ['custom', 'algorithm', 'comment']
    id_cols = ['update_time', 'create_time', 'binning_data', 'time_offset', 'deviceuuid', 'pkg_name', 'datauuid']

    def play_with_stress_data(self):
        df = self.get_formatted_df()
        if len(df) == 0:
            return df

        # when I should never sleep
        midday = 14
        time_zone_delay = 2
        ser_h = df['start_time'].dt.hour + time_zone_delay - midday
        ser_h = np.where(ser_h < 0, ser_h + 24, ser_h) + midday - 24
        df['start_time_time'] = ser_h + df['start_time'].dt.minute / 60

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.tight_layout()
        df.plot(kind='line', x='start_time', y=['min', 'max'], ax=ax1)
        # ax1.hlines(y=[-1, 7], xmin=0, xmax=len(df), colors=['r', 'r'])

        df.plot(kind='bar', x='start_time', y='score', ax=ax2, color='g')
        return df

    @staticmethod
    def simplify_column(col_name):
        if col_name.startswith('com.samsung.health.sleep.'):
            col_name = 's.' + col_name[25:]
        return col_name


class Exercise(HealthDataTable):
    csv_data_name = 'exercise'

    primary_columns = ['total_calorie', 'heart_rate_sample_count', 's.duration', 's.exercise_type',
                       's.min_altitude', 's.max_altitude', 's.mean_heart_rate', 's.count', 's.distance',
                       's.calorie', 's.mean_speed', 's.altitude_gain', 's.sweat_loss', 's.min_heart_rate',
                       's.max_heart_rate', 's.max_speed', 's.mean_cadence', 's.max_cadence',
                       's.decline_distance', 's.vo2_max', 's.incline_distance', 'source_type',
                       'reward_status', 's.count_type']
    secondary_numeric_columns = []
    secondary_str_columns = ['s.comment']
    empty_cols = ['mission_value', 'subset_data', 'routine_datauuid', 'pace_info_id', 'pace_live_data']
    id_cols = ['live_data_internal', 'sensing_status', 'location_data_internal', 'additional_internal',
               's.location_data', 's.live_data']

    @staticmethod
    def create_kml(df_kml, name, color="ff0000ff"):
        kml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'samsung_health_galaxy5_watch_data',
                                             'kmls', name))
        cols = ['longitude', 'latitude', 'altitude']
        cols = [col for col in cols if col in df_kml.columns]
        df = df_kml[cols]
        with open(kml_path, 'w') as kml_file:
            kml_file.write("<kml><Document><Placemark>" \
                           "<name>route</name>" \
                           "<visibility>0</visibility>" \
                           "<Style><LineStyle>" \
                           "<color>" + color + "</color>" \
                           "<width>3</width>" \
                           "</LineStyle></Style>" \
                           "<styleUrl>#inline600</styleUrl>" \
                           "<LineString><tessellate>1</tessellate><coordinates>")

            df.to_csv(kml_file, header=False, index=False, lineterminator=' ')
            kml_file.write("</coordinates></LineString></Placemark></Document></kml>")


    def play_with_exercise_data(self):
        # mean_speed is in m/s
        df = self.get_formatted_df()
        if len(df) == 0:
            return df
        # df_walk = df[df['s.exercise_type'] == 1001]
        # df_run = df[df['s.exercise_type'] == 1002]
        # df_pull_ups = df[df['s.exercise_type'] == 10005]
        # df_cycling = df[df['s.exercise_type'] == 11007]
        # df_hike = df[df['s.exercise_type'] == 13001]
        # df_swim_outdoor = df[df['s.exercise_type'] == 14001]
        # df_weight_machines = df[df['s.exercise_type'] == 15002]
        # df_push_up = df[df['s.exercise_type'] == 10004]
        # df_badminton = df[df['s.exercise_type'] == 6003]
        # df_handball = df[df['s.exercise_type'] == 4005]
        type_map = {1001: 'walk', 1002: 'run', 10005: 'pull_ups', 11007: 'cycling', 13001: 'hike',
                    14001: 'swim_outdoor', 15002: 'weight_machines', 10004: 'push_up', 6003: 'badminton',
                    4005: 'handball'}
        df_map = {value: df[df['s.exercise_type'] == key] for key, value in type_map.items()}

        df_run = df_map['run']
        df_run.loc[:, 's.mean_speed_kmph'] = df_run['s.mean_speed'] * 3.6
        df_run.loc[:, 's.max_speed_kmh'] = df_run['s.max_speed'] * 3.6 - df_run['s.mean_speed_kmph']

        df_run.loc[:, 's.start_time_date'] = df_run['s.start_time'].dt.date
        # df_run.plot(kind='bar', x='s.start_time_date', y='s.max_speed_kmh')
        df_plot = df_run[['s.mean_speed_kmph', 's.max_speed_kmh', 's.start_time_date']]
        df_plot.set_index('s.start_time_date', inplace=True)
        df_plot.plot.bar(stacked=True)

        df_hike = df_map['hike']
        line_ind = len(df_hike) - 1

        for i in range(len(df_hike)):
            df_location_data = self.get_json_data(df_hike['s.location_data'].iloc[i], 's.exercise')
            self.create_kml(df_location_data, "hike_" + str(df_hike['s.start_time'].dt.date.iloc[i]) + ".kml")

        df_run = df_map['run']
        for i in range(len(df_run)):
            if not df_run['s.location_data'].isna().iloc[i]:
                df_location_data = self.get_json_data(df_run['s.location_data'].iloc[i], 's.exercise')
                self.create_kml(df_location_data, "run_" + str(df_run['s.start_time'].dt.date.iloc[i]) + ".kml", "ffffff00")

        df_walk = df_map['walk']
        for i in range(len(df_walk)):
            if not df_walk['s.location_data'].isna().iloc[i]:
                df_location_data = self.get_json_data(df_walk['s.location_data'].iloc[i], 's.exercise')
                self.create_kml(df_location_data, "walk_" + str(df_walk['s.start_time'].dt.date.iloc[i]) + ".kml")

        df_bike = df_map['cycling']
        for i in range(len(df_bike)):
            if not df_bike['s.location_data'].isna().iloc[i]:
                df_location_data = self.get_json_data(df_bike['s.location_data'].iloc[i], 's.exercise')
                self.create_kml(df_location_data, "bike_" + str(df_bike['s.start_time'].dt.date.iloc[i]) + ".kml", "ffff00ff")

        # df_live_data_internal = self.get_json_data(df_hike['live_data_internal'].iloc[line_ind], 's.exercise')
        # # df_sensing_status = self.get_json_data(df_run['sensing_status'].iloc[line_ind], 's.exercise')
        # # df_location_data_internal = self.get_json_data(df_run['location_data_internal'].iloc[line_ind], 's.exercise')
        # # df_addit_internal = self.get_complex_json_data(df_run['additional_internal'].iloc[line_ind], 's.exercise')
        # df_location_data = self.get_json_data(df_hike['s.location_data'].iloc[line_ind], 's.exercise')
        # df_live_data = self.get_json_data(df_hike['s.live_data'].iloc[line_ind], 's.exercise')
        # self.create_kml(df_location_data, str(df_hike['s.start_time'].dt.date.iloc[line_ind]) + ".kml")
        # # df_datauuid = get_json_data(df_run['s.e.datauuid'].iloc[line_ind] + '.heart_rate.json', True)
        # 
        # df_location_data.plot(x='longitude', y='latitude')
        # df_live_data_internal.plot(x='elapsed_time', y='interval')
        # df_live_data.sort_values(by=['distance'], inplace=True)
        # df_live_data.reset_index(drop=True, inplace=True)
        # df_live_data.plot(x='distance', y=['cadence', 'speed'])
        return df_map

    @staticmethod
    def simplify_column(col_name):
        if col_name.startswith('com.samsung.health.exercise.'):
            col_name = 's.' + col_name[28:]
        return col_name


class HeartRate(HealthDataTable):
    csv_data_name = 'tracker.heart_rate'

    primary_columns = ['s.min', 's.max', 's.heart_rate', 's.heart_beat_count']
    secondary_str_columns = ['s.comment']
    meaningless_col = []
    empty_cols = ['source', 's.custom']
    id_cols = ['tag_id', 's.binning_data', 's.update_time', 's.create_time',
               's.time_offset', 's.deviceuuid']

    def play_with_heart_rate_data(self):
        df = self.get_formatted_df()
        df['diff'] = df['s.max'] - df['s.min']
        df['s.start_time_date'] = df['s.start_time'].dt.date

        fig, ax = plt.subplots(1, 1)
        fig.tight_layout()
        # df.plot(kind='bar', x='s.h.start_time_date', y='diff', bottom=df['s.min'])
        df.plot(x='s.start_time', y=['s.heart_rate', 's.min', 's.max'], ax=ax)
        ax.hlines(y=50, xmin=df['s.start_time'].iloc[0], xmax=df['s.start_time'].iloc[len(df)-1], colors='r')
        return df

    @staticmethod
    def simplify_column(col_name):
        if col_name.startswith('com.samsung.health.heart_rate.'):
            col_name = 's.' + col_name[30:]
        return col_name


class Weight(HealthDataTable):
    csv_data_name = 'weight'

    index_col = 'start_time'
    dates_cols = [index_col, 'update_time', 'create_time']
    primary_columns = ['skeletal_muscle', 'skeletal_muscle_mass', 'fat_free', 'fat_free_mass', 'body_fat',
                       'total_body_water', 'body_fat_mass', 'height', 'weight', 'basal_metabolic_rate']
    secondary_str_columns = []
    meaningless_col = []
    empty_cols = ['vfa_level', 'custom', 'muscle_mass']
    id_cols = ['datauuid']

    prefix = ''

    def plot_with_ma(self, df, label: str, ax, color):
        ax2 = ax.twinx()
        new_label = label + '_ma'
        df[new_label] = df[label].rolling(window=7).mean()
        df.plot(x=self.index_col, y=[label + '_ma'], ax=ax2, color=color)

    def play_with_weight_data(self):
        df = self.get_formatted_df()
        df['s_body_fat_mass'] = df['body_fat_mass'] + 30
        df['s_weight'] = df['weight'] - 30
        fig, ax = plt.subplots(1, 1)
        fig.tight_layout()
        # df.plot(kind='bar', x='s.h.start_time_date', y='diff', bottom=df['s.min'])
        # df.plot(x=self.index_col, y=['skeletal_muscle_mass', 's_body_fat_mass', 'total_body_water', 's_weight'],
        #         ax=ax)

        self.plot_with_ma(df, 'skeletal_muscle_mass', ax, 'b')
        self.plot_with_ma(df, 's_body_fat_mass', ax, 'r')

        # ax2 = ax.twinx()
        #
        # df.plot(x=self.index_col, y=['skeletal_muscle_mass'], ax=ax)
        # df.plot(x=self.index_col, y=['s_body_fat_mass'], ax=ax2, color='r')
        # ax.hlines(y=50, xmin=df['start_time'].iloc[0], xmax=df['start_time'].iloc[len(df)-1], colors='r')
        return df

    @staticmethod
    def simplify_column(col_name: str) -> str:
        return col_name


class Rewards(HealthDataTable):
    csv_data_name = 'rewards'

    index_col = 'start_time'
    dates_cols = [index_col, 'end_time', 'update_time', 'create_time']
    # primary_columns = ['skeletal_muscle', 'skeletal_muscle_mass', 'fat_free', 'fat_free_mass', 'body_fat',
    #                    'total_body_water', 'body_fat_mass', 'height', 'weight', 'basal_metabolic_rate']
    # secondary_str_columns = []
    # meaningless_col = []
    # empty_cols = ['vfa_level', 'custom', 'muscle_mass']
    id_cols = ['datauuid']

    def play_with_rewards_data(self):
        df = self.get_formatted_df()
        df['s_body_fat_mass'] = df['body_fat_mass'] + 30
        df['s_weight'] = df['weight'] - 30
        fig, ax = plt.subplots(1, 1)
        fig.tight_layout()
        # df.plot(kind='bar', x='s.h.start_time_date', y='diff', bottom=df['s.min'])
        df.plot(x=self.index_col, y=['skeletal_muscle_mass', 's_body_fat_mass', 'total_body_water', 's_weight'], ax=ax)
        # ax.hlines(y=50, xmin=df['start_time'].iloc[0], xmax=df['start_time'].iloc[len(df)-1], colors='r')
        return df

    @staticmethod
    def simplify_column(col_name: str) -> str:
        return col_name


def try_to_explain(df_combined, big_leaps=True, shift_size=1):
    if big_leaps:
        # TODO
        pass

        # try_to_explain(df_combined, big_leaps=False)
    weight_rel_cols = ['skeletal_muscle_mass', 'body_fat_mass', 'weight']
    weight_rel = df_combined[weight_rel_cols]
    returns = weight_rel - weight_rel.shift(1)

    shifted_returns = returns.shift( - shift_size)
    df_expl = df_combined.loc[:, df_combined.columns.difference(weight_rel_cols)]
    df_combined = pd.merge(shifted_returns, df_expl, left_index=True, right_index=True, how='outer')
    df_combined.dropna(subset=weight_rel_cols, inplace=True)
    shifted_returns2 = df_combined[weight_rel_cols]

    df_expl = df_combined.loc[:, df_combined.columns.difference(weight_rel_cols)]
    df_expl.fillna(0, inplace=True)
    df_expl['const'] = 1

    mmult = df_expl.T.dot(df_expl)
    mmult_inv = pd.DataFrame(np.linalg.pinv(mmult.values), mmult.columns, mmult.index)
    subject = df_expl.dot(mmult_inv)
    
    filled_returns = shifted_returns2.fillna(0)
    res = filled_returns.T.dot(subject)
    # res.rename({x: x + '_' + str(shift_size) for x in res.index}, axis=0, inplace=True)
    return res
    

def play_with_combined_data(df_weight, df_exercise, df_sleep, df_heart_rate, df_nutrients):
    time = pd.to_datetime(df_weight['start_time'])
    assert all(pd.to_datetime(df_weight['start_time']).dt.date == pd.to_datetime(df_weight['update_time']).dt.date)
    assert all(pd.to_datetime(df_weight['start_time']).dt.date == pd.to_datetime(df_weight['create_time']).dt.date)

    assert all(pd.to_datetime(df_exercise['s.start_time']).dt.date == pd.to_datetime(df_exercise['s.end_time']).dt.date)
    # assert all(pd.to_datetime(df_sleep['s.start_time']).dt.date == pd.to_datetime(df_sleep['s.end_time']).dt.date)
    assert all(pd.to_datetime(df_heart_rate['s.start_time']).dt.date ==
               pd.to_datetime(df_heart_rate['s.end_time']).dt.date)

    df_sleep['time'] = pd.to_datetime(df_sleep['s.end_time']).dt.date
    df_combined = df_sleep[['time', 'mental_recovery', 'physical_recovery', 'sleep_score', 'efficiency']] #,
                            #'sleep_duration']]
    df_combined.drop(['mental_recovery', 'physical_recovery', 'efficiency'], inplace=True, axis=1)
    df_combined = df_combined.groupby(['time']).sum()

    df_nutrients = df_nutrients[df_nutrients['calorie(k)'] > 1675]
    df_nutrients.drop(['calorie(k)', 'zinc(mg)', 'saturated fat(g)', 'magnesium(mg)', 'sugar(g)', 'fiber(g)', 'salt(g)'], inplace=True, axis=1)

    df_combined = pd.merge(df_nutrients, df_combined, left_index=True, right_index=True, how='outer')

    df_exercise['time'] = pd.to_datetime(df_exercise['s.end_time']).dt.date
    df_exercise_new = df_exercise[['time', 'total_calorie']]  # , 'heart_rate_sample_count'
    df_exercise_new = df_exercise_new.groupby(['time']).sum()
    # df_exercise_new.set_index('time', inplace=True)

    df_combined = pd.merge(df_exercise_new, df_combined, left_index=True, right_index=True, how='outer')

    df_weight['time'] = pd.to_datetime(df_weight['start_time']).dt.date
    df_weight_new = df_weight[['time', 'skeletal_muscle_mass', 'body_fat_mass', 'weight']]
    df_weight_new = df_weight_new.groupby(['time']).mean()
    # df_weight_new.set_index('time', inplace=True)

    df_combined = pd.merge(df_weight_new, df_combined, left_index=True, right_index=True, how='outer')

    # df1 = try_to_explain(df_combined, big_leaps=False, shift_size=1)
    # df2 = try_to_explain(df_combined, big_leaps=False, shift_size=2)
    # df3 = try_to_explain(df_combined, big_leaps=False, shift_size=3)
    # df4 = try_to_explain(df_combined, big_leaps=False, shift_size=4)

    results = [try_to_explain(df_combined, big_leaps=False, shift_size=shift_size) for shift_size in range(1, 10)]
    df = pd.concat(results)
    average = sum(results) / len(results)
    
    xcv = 111


def get_input_nutrients(file_name) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    df.drop(['Unnamed: 0'], inplace=True, axis=1)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df.set_index('date', inplace=True)
    return df


def main():
    direct = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'samsung_health_galaxy5_watch_data', 'data'))

    # sleep = Sleep(direct)
    # sleep.play_with_sleep_data()

    exercise = Exercise(direct)
    exercise.play_with_exercise_data()
    #
    # heart_rate = HeartRate(direct)
    # heart_rate.play_with_heart_rate_data()
    #
    # weight = Weight(direct)
    # weight.play_with_weight_data()
    #
    # stress = Stress(direct)
    # stress.play_with_stress_data()

    # rewards = Rewards(direct)
    # rewards.play_with_rewards_data()
    #
    # plt.show()

    # input_nutrients = get_input_nutrients()

    # play_with_combined_data(weight.get_formatted_df(), exercise.get_formatted_df(), sleep.get_formatted_df(),
    #                         heart_rate.get_formatted_df(), input_nutrients)


if __name__ == '__main__':
    # rename_files("data")
    main()
