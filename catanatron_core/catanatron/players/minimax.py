from catanatron import Player, state
from catanatron_experimental.cli.cli_players import register_player
from catanatron.models.player import SimplePlayer, Color
from catanatron.state_functions import (
    get_longest_road_length,
    get_played_dev_cards,
    get_player_buildings,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)

from catanatron_gym.features import (
    build_production_features,
    reachability_features,
    resource_hand_features,
)

from catanatron.models.enums import RESOURCES, BuildingType
import math



@register_player("FOO")
class FooPlayer(Player):

    DEFAULT_WEIGHTS = {
    # Where to place. Note winning is best at all costs
    "public_vps": 3e14,
    "production": 1e8,
    "enemy_production": -1e8,
    "num_tiles": 1,
    # Towards where to expand and when
    "reachable_production_0": 0,
    "reachable_production_1": 1e4,
    "buildable_nodes": 1e3,
    "longest_road": 10,
    # Hand, when to hold and when to use.
    "hand_synergy": 1e2,
    "hand_resources": 1,
    "discard_penalty": -5,
    "hand_devs": 10,
    "army_size": 10.1,
}
    
    def decide(self, game, playable_actions):
        """Should return one of the playable_actions.

        Args:
            game (Game): complete game state. read-only.
            playable_actions (Iterable[Action]): options to choose from
        Return:
            action (Action): Chosen element of playable_actions
        """
        # ===== YOUR CODE HERE =====
        # As an example we simply return the first action:

        depth = 1
        scores = {}
        for action in game.state.playable_actions:
            print(game.state.playable_actions)
            s = self.successorFunc(game, action)
            scores[action] = self.min_value(s, depth, s.state.playable_actions)
            print(scores)
        return max(scores, key=scores.get)

    def min_value(self, game, depth, playable_actions):
        #leader_actions = self.most_points(game)
        if len(game.state.playable_actions) == 0:
            print('No actions')
            #print(self.value_function(game))
            return(self.value_function(game))
        min_score = 100000
        if game.state.current_color() == self.color:
            #print('Always')
            scores = [min_score]
            for action in game.state.playable_actions:
                #print('min calling max')
                scores.append(self.max_value(self.successorFunc(game, action), depth, playable_actions))
                min_score = min(scores)
        else: 
            print('Never')
            scores = [min_score]
            for action in game.state.playable_actions:
                #print('min calling min')
                scores.append(self.min_value(self.successorFunc(game, action), depth, playable_actions))
                min_score = min(scores)
        return min_score

    def max_value(self, game, depth, playable_actions):
        #pacman_actions = gameState.getLegalActions(0)
        #print('IN MAX')
        if depth == 2 or len(game.state.playable_actions) == 0:
            #print('IN IF')
            return(self.value_function(game))
        scores = []
        for action in game.state.playable_actions:
            #print('max calling min')
            scores.append(self.min_value(self.successorFunc(game, action), depth + 1, playable_actions))
        
        best_score = max(scores)
        return best_score

    def successorFunc(self, game, action):
            #print(game)
            temp = game.copy()
            temp.execute(action)
            return temp

    def base_fn(self, params=DEFAULT_WEIGHTS):
        def fn(game, p0_color):
            production_features = build_production_features(True)
            our_production_sample = production_features(game, p0_color)
            enemy_production_sample = production_features(game, p0_color)
            production = self.value_production(our_production_sample, "P0")
            enemy_production = self.value_production(enemy_production_sample, "P1", False)

            key = player_key(game.state, p0_color)
            longest_road_length = get_longest_road_length(game.state, p0_color)

            reachability_sample = reachability_features(game, p0_color, 2)
            features = [f"P0_0_ROAD_REACHABLE_{resource}" for resource in RESOURCES]
            reachable_production_at_zero = sum([reachability_sample[f] for f in features])
            features = [f"P0_1_ROAD_REACHABLE_{resource}" for resource in RESOURCES]
            reachable_production_at_one = sum([reachability_sample[f] for f in features])

            hand_sample = resource_hand_features(game, p0_color)
            features = [f"P0_{resource}_IN_HAND" for resource in RESOURCES]
            distance_to_city = (
                max(2 - hand_sample["P0_WHEAT_IN_HAND"], 0)
                + max(3 - hand_sample["P0_ORE_IN_HAND"], 0)
            ) / 5.0  # 0 means good. 1 means bad.
            distance_to_settlement = (
                max(1 - hand_sample["P0_WHEAT_IN_HAND"], 0)
                + max(1 - hand_sample["P0_SHEEP_IN_HAND"], 0)
                + max(1 - hand_sample["P0_BRICK_IN_HAND"], 0)
                + max(1 - hand_sample["P0_WOOD_IN_HAND"], 0)
            ) / 4.0  # 0 means good. 1 means bad.
            hand_synergy = (2 - distance_to_city - distance_to_settlement) / 2

            num_in_hand = player_num_resource_cards(game.state, p0_color)
            discard_penalty = params["discard_penalty"] if num_in_hand > 7 else 0

            # blockability
            buildings = game.state.buildings_by_color[p0_color]
            owned_nodes = buildings[BuildingType.SETTLEMENT] + buildings[BuildingType.CITY]
            owned_tiles = set()
            for n in owned_nodes:
                owned_tiles.update(game.state.board.map.adjacent_tiles[n])
            num_tiles = len(owned_tiles)

            # TODO: Simplify to linear(?)
            num_buildable_nodes = len(game.state.board.buildable_node_ids(p0_color))
            longest_road_factor = (
                params["longest_road"] if num_buildable_nodes == 0 else 0.1
            )

            return float(
                game.state.player_state[f"{key}_VICTORY_POINTS"] * params["public_vps"]
                + production * params["production"]
                + enemy_production * params["enemy_production"]
                + reachable_production_at_zero * params["reachable_production_0"]
                + reachable_production_at_one * params["reachable_production_1"]
                + hand_synergy * params["hand_synergy"]
                + num_buildable_nodes * params["buildable_nodes"]
                + num_tiles * params["num_tiles"]
                + num_in_hand * params["hand_resources"]
                + discard_penalty
                + longest_road_length * longest_road_factor
                + player_num_dev_cards(game.state, p0_color) * params["hand_devs"]
                + get_played_dev_cards(game.state, p0_color, "KNIGHT") * params["army_size"]
            )
        return fn

    def value_production(self, sample, player_name="P0", include_variety=True):
        proba_point = 2.778 / 100
        features = [
            f"EFFECTIVE_{player_name}_WHEAT_PRODUCTION",
            f"EFFECTIVE_{player_name}_ORE_PRODUCTION",
            f"EFFECTIVE_{player_name}_SHEEP_PRODUCTION",
            f"EFFECTIVE_{player_name}_WOOD_PRODUCTION",
            f"EFFECTIVE_{player_name}_BRICK_PRODUCTION",
        ]
        #print('SAMPLE ' + str(sample))
        prod_sum = 0
        #prod_sum = sum([sample[f] for f in features])
        for f in features:
            prod_sum += sample[f]
        TRANSLATE_VARIETY = 4 
        prod_variety = (
            sum([sample[f] != 0 for f in features]) * TRANSLATE_VARIETY * proba_point
        )
        return prod_sum + (0 if not include_variety else prod_variety)

    def value_function(self, game):
        value_fn = self.base_fn()
        value = value_fn(game, self.color)
        print(value)
        return value

    def most_points(self, game):
        game_keys = game.state.player_state
        my_key = state.player_key(game.state, self.color)
        white_key = state.player_key(game.state, Color.WHITE)
        red_key  = state.player_key(game.state, Color.RED)
        orange_key = state.player_key(game.state, Color.ORANGE)
        blue_key = state.player_key(game.state, Color.BLUE)
        player_keys = [white_key, red_key, orange_key, blue_key]
        max_points = 0
        max_player = None
        for p in player_keys:
            items = dict(filter(lambda item: p in item[0], game_keys.items()))
            player_points  = list(items.values())[0]
            print(player_points)
            if player_points >= max_points:
                max_points = player_points
                max_player = p
                
        print('Max player ' + str(max_player) + ' with points: ' + str(max_points))
        return max_player


    


    
    
        self.getAction(game)
        return playable_actions[0]
    # ===== END YOUR CODE =====