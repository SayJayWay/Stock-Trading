import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

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

df['Output_1'] = np.where(df['Adj_Close_lead_5'] > df['Adj Close'], 0.99, 0.01)
df['Output_2'] = np.where(df['Adj_Close_lead_5'] <= df['Adj Close'], 0.99, 0.01)

df.dropna(inplace = True)
df.reset_index(inplace = True)

inputs = df[['MACD_lag_1', 'MACD_lag_2', 'MACD_lag_3',
             'Open_lag_1', 'Open_lag_2', 'Open_lag_3', 'High_lag_1',
             'High_lag_2', 'High_lag_3', 'Low_lag_1', 'Low_lag_2', 'Low_lag_3',
             'Adj_Close_lag_1', 'Adj_Close_lag_2', 'Adj_Close_lag_3']]

outputs = df[['Output_1', 'Output_2']]

learning_rate = 0.00001
batch_size = 2
update_step = 7

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

cf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=ntwk_output_1, labels = target_output))

ts = tf.train.GradientDescentOptimizer(learning_rate).minimize(cf)

cp = tf.equal(tf.argmax(ntwk_output_2, 1), tf.argmax(outputs))

acc = tf.reduce_mean(tf.cast(cp, tf.float32))

def next_batch(num, data, labels):
    '''
    Return a total of 'num' random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    # print(idx)
    data_shuffle = [data.iloc[i] for i in idx]
    labels_shuffle = [labels.iloc[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_epochs = 1
    for epoch in range(num_epochs):
        total_cost = 0
        for _ in range(int(len(df['Adj Close']) / batch_size)):
            batch_x, batch_y = next_batch(batch_size, inputs, outputs)
            t, c = sess.run([ts, cf], feed_dict = {network_input: batch_x, target_output: batch_y})
            total_cost += c
            print('Epoch', epoch, 'completed out of', num_epochs, 'loss:', total_cost)
        
        test_x, test_y = next_batch(batch_size, inputs, outputs)
        print('Accuracy:', acc.eval({network_input:test_x, target_output: test_y}))