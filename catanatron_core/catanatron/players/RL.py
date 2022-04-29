import os
import time

import gym
import numpy as np
import tqdm as tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Activation,
    Flatten,
    BatchNormalization,
)
import tensorflow as tf


from catanatron.models.player import Player
from catanatron.models.enums import Action, ActionType
from catanatron_experimental.cli.cli_players import register_player
from catanatron_gym.features import (
    create_sample,
    create_sample_vector,
    get_feature_ordering,
)
from catanatron_gym.envs.catanatron_env import ACTIONS_ARRAY, ACTION_SPACE_SIZE
from catanatron_experimental.machine_learning.board_tensor_features import (
    NUMERIC_FEATURES,
    create_board_tensor,
)
from collections import Counter, deque
import random
from tqdm import tqdm


# from catanatron_experimental.rep_b_model import build_model

# Taken from correlation analysis
from catanatron_experimental.machine_learning.players.reinforcement import hot_one_encode_action
from catanatron_gym.envs.catanatron_env import NUM_FEATURES

from catanatron_gym.envs.catanatron_env import to_action_space, from_action_space


class DQN:
    def __init__(self):
        self.model = self.create_model()
        self.target_update_counter = 0

    def create_model(self):
        inputs = tf.keras.Input(shape=(NUM_FEATURES,))
        outputs = inputs
        self.replay_memory = deque(maxlen=5000)

        # mean = np.load(NORMALIZATION_MEAN_PATH)
        # variance = np.load(NORMALIZATION_VARIANCE_PATH)
        # normalizer_layer = Normalization(mean=mean, variance=variance)
         #outputs = normalizer_layer(outputs)

        # outputs = Dense(8, activation="relu")(outputs)
        outputs = Dense(units=ACTION_SPACE_SIZE, activation="linear")(outputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["mae"])
        return model
# Trains main network every step during episode
    def train(self, terminal_state):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < 10000:
            print("Not enough training data", len(self.replay_memory), 1000)
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, 1000)

        # Get current states from minibatch, then query NN model for Q values
        current_states = tf.convert_to_tensor([t[0][0] for t in minibatch])
        current_qs_list = self.model.call(current_states).numpy()

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = tf.convert_to_tensor([t[3][0] for t in minibatch])
        future_qs_list = self.target_model.call(new_current_states).numpy()

        # Create X, y for training
        X = []
        y = []
        action_ints = list(map(lambda b: b[1], minibatch))
        action_ints_counter = Counter(action_ints)
        sample_weight = []
        for index, (
            current_state,
            action,
            reward,
            new_current_state,
            done,
        ) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
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
            # w = MINIBATCH_SIZE / (
            #     len(action_ints_counter) * action_ints_counter[action]
            # )
            w = 1 / (action_ints_counter[action])
            sample_weight.append(w)

        # print("Training at", len(self.replay_memory), MINIBATCH_SIZE)
        # print(action_ints_counter)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(
            tf.convert_to_tensor(X),
            tf.convert_to_tensor(y),
            sample_weight=np.array(sample_weight),
            batch_size=1000,
            epochs=1,
            verbose=0,
            shuffle=False,  # no need since minibatch already was a random sampling
        )

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > 5:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            print("Updated model!")

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        (sample) = state
        sample = tf.reshape(tf.convert_to_tensor(sample), (-1, NUM_FEATURES))
        return self.model.call(sample)[0]


FEATURES = [
    "P0_HAS_ROAD",
    "P1_HAS_ROAD",
    "P0_HAS_ARMY",
    "P1_HAS_ARMY",
    "P0_ORE_PRODUCTION",
    "P0_WOOD_PRODUCTION",
    "P0_WHEAT_PRODUCTION",
    "P0_SHEEP_PRODUCTION",
    "P0_BRICK_PRODUCTION",
    "P0_LONGEST_ROAD_LENGTH",
    "P1_ORE_PRODUCTION",
    "P1_WOOD_PRODUCTION",
    "P1_WHEAT_PRODUCTION",
    "P1_SHEEP_PRODUCTION",
    "P1_BRICK_PRODUCTION",
    "P1_LONGEST_ROAD_LENGTH",
    "P0_PUBLIC_VPS",
    "P1_PUBLIC_VPS",
    "P0_SETTLEMENTS_LEFT",
    "P1_SETTLEMENTS_LEFT",
    "P0_CITIES_LEFT",
    "P1_CITIES_LEFT",
    "P0_KNIGHT_PLAYED",
    "P1_KNIGHT_PLAYED",
]


def allow_feature(feature_name):
    return (
            "2_ROAD" not in feature_name
            and "HAND" not in feature_name
            and "BANK" not in feature_name
            and "P0_ACTUAL_VPS" != feature_name
            and "PLAYABLE" not in feature_name
            and "LEFT" not in feature_name
            and "ROLLED" not in feature_name
            and "PLAYED" not in feature_name
            and "PUBLIC_VPS" not in feature_name
            and not ("TOTAL" in feature_name and "P1" in feature_name)
            and not ("EFFECTIVE" in feature_name and "P0" in feature_name)
            and (feature_name[-6:] != "PLAYED" or "KNIGHT" in feature_name)
    )


ALL_FEATURES = get_feature_ordering(num_players=2)
FEATURES = list(filter(allow_feature, ALL_FEATURES))
FEATURES = get_feature_ordering(2)
FEATURE_INDICES = [ALL_FEATURES.index(f) for f in FEATURES]

EPSILON = 0.20
Q_MODEL = None


def q_model_path(version):
    return os.path.join(os.path.dirname(__file__), "q_models", str(version))


@register_player("Q")
class RLPlayer(Player):
       # self.model_path = model_path
    global Q_MODEL
    #Q_MODEL = DQN().create_model()
    Q_MODEL = keras.lo
    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        sample = create_sample_vector(game, self.color, FEATURES)
        sample = tf.reshape(tf.convert_to_tensor(sample), (-1, NUM_FEATURES))

        qs = Q_MODEL.call(sample)[0]

        best_action_int = epsilon_greedy_policy(playable_actions, qs, 0.05)
        best_action = from_action_space(best_action_int, playable_actions)
        return best_action


def epsilon_greedy_policy(playable_actions, qs, epsilon):
    if np.random.random() > epsilon:
        # Create array like [0,0,1,0,0,0,1,...] representing actions in space that are playable
        action_ints = list(map(to_action_space, playable_actions))
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int)
        mask[action_ints] = 1

        clipped_probas = np.multiply(mask, qs)
        clipped_probas[clipped_probas == 0] = -np.inf

        best_action_int = np.argmax(clipped_probas)
    else:
        # Get random action
        index = random.randrange(0, len(playable_actions))
        best_action = playable_actions[index]
        best_action_int = to_action_space(best_action)

    return best_action_int

def train():
    epsilon = EPSILON
    env = gym.make("catanatron_gym:catanatron-v0")

    random.seed(3)
    tf.random.set_seed(3)

    model_name = "DQN - {}".format(time.time())
    model_folder = "data/models/"
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)

    DeepQ = DQN()
    metric = "data/logs/catan-dnq"
    output_path = model_folder + model_name

    for episode in tqdm(range(1, 10000 + 1 ), ascii = True, unit="episodes"):
        ep_rewards = []
        ep_reward = 0;
        step = 1

        observation = env.reset()

        done = False
        while not done:
            best_action_int = epsilon_greedy_policy(env.get_playable_actions(),
                                                    DeepQ.get_qs(observation), EPSILON)
            new_state, reward, done, info = env.step(best_action_int)

            ep_reward += reward

            DeepQ.update_replay_memory((observation, best_action_int, reward, new_state, done))

            if step % 50 == 0:
                DeepQ.train(done)

            observation = new_state
            step += 1
        if step % 50 == 0:
            DeepQ.train(done)

            # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(ep_reward)
        if episode % 10 == 0:
            average_reward = sum(ep_rewards[-10:]) / len(ep_rewards[-10:])
            # with writer.as_default():
            #     tf.summary.scalar("avg-reward", average_reward, step=episode)
            #     tf.summary.scalar("epsilon", epsilon, step=episode)
            #     writer.flush()

        # Decay epsilon
        if epsilon > 0.001:
            epsilon *= 0.99975
            epsilon = max(0.001, epsilon)

    print("Saving model to", output_path)
    DeepQ.model.save(output_path)


if __name__ == "__main__":
    train()
### Training session
