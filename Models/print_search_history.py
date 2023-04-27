import numpy as np
import os

path_to_search_history = './visual_wake_word_search_history/learning_rate_0.001/batch_size_16'

def load_search_history_as_strctured_np_array(path_to_search_history='./search_history') :
    dtype = np.dtype([('k', 'i'), ('c', 'i'), ('history', 'O')])
    training_histories = np.array([], dtype=dtype)
    for root, dirs, files in os.walk(path_to_search_history):
        for file in files:
            if file.endswith('_hist.npy'):
                keys = file.split('_')
                k = int(keys[0][1:])
                c = int(keys[1][1:])
                training_history = np.load(os.path.join(root, file), allow_pickle=True)
                training_histories = np.append(training_histories, np.array((k, c, training_history), dtype=dtype))
    return training_histories

search_history = load_search_history_as_strctured_np_array(path_to_search_history)

search_history = np.sort(search_history, order=['k', 'c'])

for training_history in search_history :
    print(f"k: {training_history['k']}, c: {training_history['c']}, max val acc until epoch 3: {np.round(np.amax(training_history['history'].item().get('val_accuracy')), decimals=3)}")
