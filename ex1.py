## https://www.tensorflow.org/versions/r0.10/tutorials/wide/index.html#reading-the-census-data
## 참조할것
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd
import tempfile

class ex1_batchtrain:
    def __init__ (self, df):
        self.idf = df
        self.lw = np.arange(len(df))
    def next_batch(self,size):
        t = np.random.choice(self.lw,size)
        return [self.idf.iloc[t]]


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("train_steps", 10000, "Number of training steps.")
flags.DEFINE_string("model_type", "wide",
"Valid model types: {'wide', 'deep', 'wide_n_deep'}.")


tf.logging.set_verbosity(tf.logging.INFO) # for logging


# Data sets
TRAININGCSV = "training_set.csv"
TESTCSV = "test_set.csv"
flags.DEFINE_string("test_name","FINAL_post","Name of Test")

CATEGORICAL_COLUMNS = ['multiple', "shape", "sizeincrease"]
CONTINUOUS_COLUMNS = ["size", "surgicalage"]
LABEL_COLUMN = "label"


def setdf(dfs):
    df = dfs.copy(deep=True)
    df["age"] = df["age"] / 100.0
    df["size"] = df["size"] / 20.0
    df["BMI"] = df["BMI"] / 40.0
    df["sex"] = df["sex"].apply(forlabeling)
    df["multiple"] = df["multiple"].apply(forlabeling)
    df["shape"] = df["shape"].apply(forlabeling)
    df["sizeincrease"] = df["sizeincrease"].apply(forlabeling)
    df["stone"] = df["stone"].apply(forlabeling)
    return df

def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def build_estimator(model_dir):
    # Sparse base columns.

    sex = tf.contrib.layers.sparse_column_with_integerized_feature(column_name="sex",
                                                     bucket_size=3)
    multiple = tf.contrib.layers.sparse_column_with_integerized_feature(column_name="multiple",
                                                     bucket_size=3)
    shape = tf.contrib.layers.sparse_column_with_integerized_feature(column_name="shape",
                                                     bucket_size=3)
    sizeincrease = tf.contrib.layers.sparse_column_with_integerized_feature(column_name="sizeincrease",
                                                   bucket_size=3)
    stone = tf.contrib.layers.sparse_column_with_integerized_feature(column_name="stone",
                                                   bucket_size=3)

    # Continuous base columns.
    surgicalage = tf.contrib.layers.real_valued_column("surgicalage")
    size = tf.contrib.layers.real_valued_column("size")

    # Wide columns and deep columns.
    wide_columns = [ multiple, shape, sizeincrease]
    deep_columns = [
        tf.contrib.layers.embedding_column(multiple, dimension=5),
        tf.contrib.layers.embedding_column(shape, dimension=5),
        tf.contrib.layers.embedding_column(sizeincrease,dimension=5),
        tf.contrib.layers.embedding_column(sex, dimension=5),
        tf.contrib.layers.embedding_column(stone, dimension=5),
        surgicalage,
        size,
        ]



    if FLAGS.model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir, feature_columns=wide_columns)
    if FLAGS.model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=[9,4,2],
            optimizer=tf.train.AdamOptimizer(
      learning_rate=0.001) )
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        linear_optimizer=tf.train.FtrlOptimizer(
            learning_rate=0.001,
            l1_regularization_strength=0.1,
            l2_regularization_strength=0.1),
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[9,4,2],
        dnn_optimizer=tf.train.AdamOptimizer(
      learning_rate=0.001,epsilon=0.001)
        )
    return m
    # wide & deep column 필요시 sparse colum 정정

def forlabeling(x):
    if x == 1:
        return 0
    else:
        return 1

df_train = pd.read_csv(TRAININGCSV)
df_test  = pd.read_csv(TESTCSV)

df_train[LABEL_COLUMN] = (
  df_train["neoplastic"].apply(forlabeling))
df_test[LABEL_COLUMN] = (
  df_test["neoplastic"].apply(forlabeling))

df_train = setdf(df_train)
df_test = setdf(df_test)
model_dir = tempfile.mkdtemp()
print(model_dir)

df_test.drop(['Hb' ,'T.chole' ,'TB' ,'ALT' ,'ALP' ,'Glucose', 'HBsAg'],inplace=True,axis=1)

m = build_estimator(model_dir)

m.fit(input_fn=lambda: input_fn(df_train), steps=FLAGS.train_steps)


results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
y = m.predict(input_fn=lambda: input_fn(df_test))
df_test["prediction"] = y
ds = df_test["label"]
df_test["a"] = ((ds==y)&(ds==1)).astype(int)
df_test["b"] = ((ds!=y)&(ds==1)).astype(int)
df_test["c"] = ((ds!=y)&(ds==0)).astype(int)
df_test["d"] = ((ds==y)&(ds==0)).astype(int)
csvname = 'output_'+ FLAGS.test_name + FLAGS.model_type + ".csv"
df_test.to_csv(csvname)
