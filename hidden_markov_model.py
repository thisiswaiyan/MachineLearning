import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.8, 0,2]) # first day in our sequence has an 80% chance of being cold
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], # A cold day has a 30% chance of being followed by a hot day
                                                 [0.2, 0.8]]) # A hot day has a 20% chance of being followed by a cold day
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

# loc argument represents the mean value and the scale is the standard deviation

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

mean = model.mean()

with tf.compat.v1.Session() as sess:
    print(mean.numpy())