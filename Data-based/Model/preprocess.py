import pandas as pd
import numpy as np
import csv

def preprocessData(path_to_csv):
    data = pd.read_csv(path_to_csv)
    data['total_rooms'] = np.log(data['total_rooms'] + 1)
    data['total_bedrooms'] = np.log(data['total_bedrooms'] + 1)
    data['population'] = np.log(data['population'] + 1)
    data['households'] = np.log(data['households'] + 1)
    data = data.join(pd.get_dummies(data.ocean_proximity)).drop(['ocean_proximity'], axis=1)
    data['bedroom_ratio'] = data['total_bedrooms'] / data['total_rooms']
    data['household_rooms'] = data['total_rooms'] / data['households']
    # data.to_csv(f'preprocessed({path_to_csv}).csv', index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    data.to_json(f'preprocessed({path_to_csv}).jsonl', orient='values')
    return 'preprocessed({path_to_csv}).csv'

preprocessData('batch.csv')