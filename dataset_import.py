import tensorflow_datasets as tfds
import tensorflow as tf

# tfds works in both Eager and Graph modes
tf.compat.v1.enable_eager_execution()

# See available datasets
# print(tfds.list_builders())

# Construct a tf.data.Dataset
ds_test = tfds.load(name="voc2007", split=tfds.Split.TEST.subsplit(tfds.percent[25:75]))
print(ds_test.take(1))
