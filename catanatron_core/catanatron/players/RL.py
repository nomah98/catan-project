import os
import time

import gym
import numpy as np
import tqdm as tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Dense,
)
import tensorflow as tf

from catanatron.models.player import Player
from catanatron_experimental.cli.cli_players import register_player
from catanatron_gym.features import (
    create_sample_vector,
    get_feature_ordering,
)

from catanatron_gym.envs.catanatron_env import ACTION_SPACE_SIZE

from collections import Counter, deque
import random
from tqdm import tqdm

from catanatron_gym.envs.catanatron_env import NUM_FEATURES

from catanatron_gym.envs.catanatron_env import to_action_space, from_action_space


class DQN:
    def __init__(self):
        self.model = self.create_model()
        self.curr_count = 0

    def create_model(self):
        inputs = tf.keras.Input(shape=(NUM_FEATURES,))
        outputs = inputs
        self.past_episodes = deque(maxlen=5000)

        outputs = Dense(units=ACTION_SPACE_SIZE, activation="linear")(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["mae"])
        return model

    # Trains main network every step during episode
    def train(self, terminal_state):
        # Start training only if certain number of samples is already saved
        if len(self.past_episodes) < 10000:
            return

        # Get a subset of random samples from Q-table
        subset = random.sample(self.past_episodes, 1000)

        current_states = tf.convert_to_tensor([t[0][0] for t in subset])
        current_qs_list = self.model.call(current_states).numpy()

        # Here we use the Neural network to approximate future Q-values
        new_current_states = tf.convert_to_tensor([t[3][0] for t in subset])
        future_qs_list = self.target_model.call(new_current_states).numpy()

        X = []
        y = []

        # Feature to int taken from catanatron gym
        action_ints = list(map(lambda b: b[1], subset))
        action_ints_counter = Counter(action_ints)
        episode_weight = []
        for index, (current_state, action, reward, next_state, finished,) in enumerate(subset):

            if not finished:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + 0.9 * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state[0])
            y.append(current_qs)

            # Calculate the weights of the features
            weight = 1 / (action_ints_counter[action])
            episode_weight.append(weight)

        # Update the model
        self.model.fit(
            tf.convert_to_tensor(X),
            tf.convert_to_tensor(y),
            sample_weight=np.array(episode_weight),
            batch_size=1000,
            epochs=1,
            verbose=0,
            shuffle=False,
        )

        # Update target network counter every episode
        if terminal_state:
            self.curr_count += 1

        # If counter reaches set value, update target network with weights of main network
        if self.curr_count > 5:
            self.target_model.set_weights(self.model.get_weights())
            self.curr_count = 0
            print("Updated model!")

    def episode_update(self, transition):
        self.past_episodes.append(transition)

    # Get approximation for Q-value
    def q_state(self, state):
        (sample) = state
        sample = tf.reshape(tf.convert_to_tensor(sample), (-1, NUM_FEATURES))
        return self.model.call(sample)[0]


# Pick and choose what features to allow
def allow_feature(feature_name):
    return (
            "2_ROAD" not in feature_name
            and "HAND" not in feature_name
            and "BANK" not in feature_name
            and "LEFT" not in feature_name
            and "ROLLED" not in feature_name
            and "PLAYED" not in feature_name
            and "PUBLIC_VPS" not in feature_name
            and not ("TOTAL" in feature_name and "P1" in feature_name)
            and not ("EFFECTIVE" in feature_name and "P0" in feature_name)
            and "P0_ACTUAL_VPS" != feature_name
            and "PLAYABLE" not in feature_name
            and (feature_name[-6:] != "PLAYED" or "KNIGHT" in feature_name)
    )


ALL_FEATURES = get_feature_ordering(num_players=2)
FEATURES = list(filter(allow_feature, ALL_FEATURES))
FEATURES = get_feature_ordering(2)
FEATURE_INDICES = [ALL_FEATURES.index(f) for f in FEATURES]

EPSILON = 0.20


@register_player("Q")
class RLPlayer(Player):
    global Q_MODEL
    # Q_MODEL = DQN().create_model()
    Q_MODEL = tf.keras.models.load_model("data/models/DQN - 1651202649.4990249")

    def decide(self, game, playable_actions):
        # If there is only one action to play, play it
        if len(playable_actions) == 1:
            return playable_actions[0]

        sample = create_sample_vector(game, self.color, FEATURES)
        sample = tf.reshape(tf.convert_to_tensor(sample), (-1, NUM_FEATURES))

        qs = Q_MODEL.call(sample)[0]

        best_action_int = epsilon_greedy_policy(playable_actions, qs, 0.05)
        best_action = from_action_space(best_action_int, playable_actions)
        return best_action


def epsilon_greedy_policy(playable_actions, qs, epsilon):
    if np.random.random() <= epsilon:
        index = random.randrange(0, len(playable_actions))
        best_action = playable_actions[index]
        best_action_int = to_action_space(best_action)
    else:
        action_ints = list(map(to_action_space, playable_actions))
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int)
        mask[action_ints] = 1

        clipped_probas = np.multiply(mask, qs)
        clipped_probas[clipped_probas == 0] = -np.inf

        best_action_int = np.argmax(clipped_probas)

    return best_action_int


def train():
    epsilon = EPSILON
    env = gym.make("catanatron_gym:catanatron-v0")

    random.seed(3)
    tf.random.set_seed(3)

    model_name = "DQN - {}".format(time.time())
    model_folder = "data/models/"
    # Output the model
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)

    DeepQ = DQN()
    output_path = model_folder + model_name

    for _ in tqdm(range(1, 10000 + 1), ascii=True, unit="episodes"):
        ep_rewards = []
        ep_reward = 0;
        step = 1

        observation = env.reset()

        done = False
        while not done:
            featurized_action = epsilon_greedy_policy(env.get_playable_actions(),
                                                      DeepQ.q_state(observation), EPSILON)
            new_state, reward, done, info = env.step(featurized_action)

            ep_reward += reward

            DeepQ.episode_update((observation, featurized_action, reward, new_state, done))

            if step % 50 == 0:
                DeepQ.train(done)

            observation = new_state
            step += 1
        if step % 50 == 0:
            DeepQ.train(done)

            # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(ep_reward)

        if epsilon > 0.001:
            epsilon *= 0.99975
            epsilon = max(0.001, epsilon)

    print("Saving model to", output_path)
    DeepQ.model.save(output_path)


if __name__ == "__main__":
    train()