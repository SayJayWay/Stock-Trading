import pandas as pd
import numpy as np
import tensorflow as tf

df = pd.read_csv('../stock_dfs/AAPL.csv')

df['MA26'] = df['Adj Close'].rolling(26).mean()
df['MA12'] = df['Adj Close'].rolling(12).mean()
df['MACD'] = df['MA12'] - df['MA26']

for lag in range(1,4):
    df['MACD_lag_' + str(lag)] = df['MACD'].shift(lag)
    df['Open_lag_' + str(lag)] = df['Open'].shift(lag)
    df['High_lag_' + str(lag)] = df['High'].shift(lag)
    df['Low_lag_' + str(lag)] = df['Low'].shift(lag)
    df['Adj_Close_lag_' + str(lag)] = df['Adj Close'].shift(lag)
    
df['Adj_Close_lead_5'] = df['Adj Close'].shift(-5)

df['sign_lead_5'] = np.where(df['Adj_Close_lead_5'] > df['Adj Close'], 1, -1)

df.dropna(inplace = True)


learning_rate = 0.0001
batch_size = 100
update_tep = 10

input_nodes = 15
layer_1_nodes = 15
layer_2_nodes = 15
layer_3_nodes = 15
output_nodes = 2

network_input = tf.placeholder(tf.float32, [None, input_nodes])
target_output = tf.placeholder(tf.float32, [None, output_nodes])

layer_1 = tf.Variable(tf.random_normal([input_nodes, layer_1_nodes]))
layer_1_bias = tf.Variable(tf.random_normal([layer_1_nodes]))

layer_2 = tf.Variable(tf.random_normal([layer_1_nodes, layer_2_nodes]))
layer_2_bias = tf.Variable(tf.random_normal([layer_2_nodes]))

layer_3 = tf.Variable(tf.random_normal([layer_2_nodes, layer_3_nodes]))
layer_3_bias = tf.Variable(tf.random_normal([layer_3_nodes]))

out_layer = tf.Variable(tf.random_normal([layer_3_nodes, output_nodes]))
out_layer_bias = tf.Variable(tf.random_normal([output_nodes]))

l1_output = tf.nn.relu(tf.matmul(network_input, layer_1) + layer_1_bias)
l2_output = tf.nn.relu(tf.matmul(l1_output, layer_2) + layer_2_bias)
l3_output = tf.nn.relu(tf.matmul(l2_output, layer_3) + layer_3_bias)

ntwk_output_1 = tf.matmul(l3_output, out_layer) + out_layer_bias
ntwk_output_2 = tf.nn.softmax(ntwk_output_1)

cf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ntwk_output_1, labels = target_output))

ts = tf.train.GradientDescentOptimizer(learning_rate).minimize(cf)

cp = tf.equal(tf.argmax(ntwk_output_2, 1), tf.argmax(df['sign_lead_5']))

acc = tf.reduce_mean(tf.cast(cp, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_epochs = 10
    for epoch in range(num_epochs):
        total_cost = 0
        for _ in range(len(df['Adj Close']) / batch_size):
            batch_x, batch_y = 