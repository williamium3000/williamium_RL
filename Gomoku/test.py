import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


# files = ["data_set/gomocup/dataset_gomocup15_tfrec_a_dist/train_a_dist.tfrecords"]
# filename_queue = tf.train.string_input_producer(files)

# # if cfg.DATA_SET_SUFFIX[3:7] == 'dist':
# read_features = {'state': tf.compat.v1.FixedLenFeature([225], tf.int64),
#                              'actions': tf.compat.v1.FixedLenFeature([225], tf.float32)}

read_features = {'state': tf.compat.v1.FixedLenFeature([225], tf.int64),
                             'action': tf.compat.v1.FixedLenFeature([], tf.int64)}
def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, read_features)
# reader = tf.TFRecordReader()
# _, serialized_example = reader.read(filename_queue)
# features = tf.io.parse_single_example(serialized_example, features=read_features)
# s = features['state']
#         s = tf.cast(s, tf.float32)
#         if 'actions' in features:
#             a_dist = features['actions']
#         elif 'action' in features:
#             a_dist = tf.one_hot(features['action'], Board.BOARD_SIZE_SQ, dtype=tf.float32)
#         else:
#             raise ValueError('unknown feature')
# s, a_dist = tf.train.shuffle_batch([s, a_dist],
#                                            batch_size=128,
#                                            num_threads=1,
#                                            capacity=2000,
#                                            min_after_dequeue=2000)
# with tf.Session() as session:
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=session,coord=coord)
    
#     # if 'actions' in features:
#     #     a_dist = features['actions']
#     # elif 'action' in features:
#     #     a_dist = tf.one_hot(features['action'], 225, dtype=tf.float32)
#     # else:
#     #     raise ValueError('unknown feature')
#     print(s)
#     print(a)
filenames = ["data_set/gomocup/dataset_gomocup15_tfrec_a_proportional/validation_a_proportional.tfrecords"]
dataset = tf.data.TFRecordDataset(filenames).map(_parse_function)
# dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
# print(dataset)
# dataset = list(dataset.as_numpy_iterator())
# print(dataset)
# for e in dataset.as_numpy_iterator():
#     print(e)
# for raw_record in dataset.take(1):
#     example = tf.train.Example()
#     # example.ParseFromString(raw_record.numpy())
#     print(example)
# print(dataset)

actions = []
states = []
cnt = 0
for e in dataset:
    cnt += 1
    action = e["action"].numpy()
    state = e["state"].numpy()
    actions.append([action])
    states.append(state)
    if cnt % 10000 == 0:
        print(cnt)
actions = np.array(actions)
states = np.array(states)
np.savez("data_set/gomocup/dataset_gomocup15_tfrec_a_proportional/validation_a_proportional.npz", actions = actions, states = states)
# with tf.compat.v1.Session() as session:
#     coord = tf.train.Coordinator()
#     threads = tf.compat.v1.train.start_queue_runners(sess=session,coord=coord)
#     for e in dataset:
#         cnt += 1
#         state = e["actions"].numpy()
#         action = e["state"].numpy()
#         # np.concatenate([state, action])
#     print(cnt)