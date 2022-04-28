import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from catanatron.models.player import Player
from catanatron.models.enums import Action, ActionType
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



# from catanatron_experimental.rep_b_model import build_model

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

EPSILON = 0.20  # for epsilon-greedy action selection
# singleton for model. lazy-initialize to easy dependency graph and stored
#   here instead of class attribute to skip saving model in CatanatronDB.
Q_MODEL = None

def q_model_path(version):
    return os.path.join(os.path.dirname(__file__), "q_models", str(version))


class QRLPlayer(Player):
    def __init__(self, color, model_path):
        super(QRLPlayer, self).__init__(color)
        self.model_path = model_path
        global Q_MODEL
        Q_MODEL = keras.models.load_model(model_path)


    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        # epsilon-greedy: with EPSILON probability play at random.
        # if random.random() < EPSILON:
        #     index = random.randrange(0, len(playable_actions))
        #     return playable_actions[index]

        # Create sample matrix of state + action vectors.
        state = create_sample_vector(game, self.color, FEATURES)
        samples = []
        for action in playable_actions:
            samples.append(np.concatenate((state, self.hot_one_encode_action(action))))
        X = np.array(samples)

        # Predict on all samples
        result = Q_MODEL.predict(X)
        index = np.argmax(result)
        return playable_actions[index]

    def hot_one_encode_action(self,action):
        normalized = self.normalize_action(action)
        index = ACTIONS_ARRAY.index((normalized.action_type, normalized.value))
        vector = np.zeros(ACTION_SPACE_SIZE, dtype=int)
        vector[index] = 1
        return vector

    def normalize_action(self, action):
        normalized = action
        if normalized.action_type == ActionType.ROLL:
            return Action(action.color, action.action_type, None)
        elif normalized.action_type == ActionType.MOVE_ROBBER:
            return Action(action.color, action.action_type, action.value[0])
        elif normalized.action_type == ActionType.BUILD_ROAD:
            return Action(action.color, action.action_type, tuple(sorted(action.value)))
        elif normalized.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            return Action(action.color, action.action_type, None)
        elif normalized.action_type == ActionType.DISCARD:
            return Action(action.color, action.action_type, None)

        return normalized

