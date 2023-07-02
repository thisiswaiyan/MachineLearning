import tensorflow as tf

# rank zero tensor
string = tf.Variable("This is tensorflow", tf.string)
number = tf.Variable(123, tf.int16)
floating = tf.Variable(4.567, tf.float64)

rank1_tensor = tf.Variable(["Test", "Ok", "Tim"], tf.string) # rank 1 tensor because 1 list and 1 array
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string) # call rank 2 tensor because 2 list and 2 array

print(tf.rank(rank2_tensor))

print(rank2_tensor.shape)

# shaping
tensor1 = tf.ones([1,2,3])
tensor2 = tf.reshape(tensor1, [2,3,1])
tensor3 = tf.reshape(tensor2, [3, -1])  # -1 tells the tensor to calculate the size of the dimension in that place
                                        # this will reshape the tensor to [3,2]
print(tensor1)
print(tensor2)
print(tensor3)

#with tf.Session() as sess: # create a session using the graph
#    tensor.eval()   # tensor will of course be the name of your tensor

# %tensorflow_version 2.x
print(tf.version)
t = tf.zeros([5,5,5,5])

t = tf.reshape(t, [125, -1])
print(t)