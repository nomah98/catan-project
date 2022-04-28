

from catanatron import Player, state
from catanatron_experimental.cli.cli_players import register_player
from catanatron.models.player import SimplePlayer, Color


@register_player("FOO")
class FooPlayer(Player):
    
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
    #if game.state.player_state.keys
    player_states = []
    game_keys = game.state.player_state
    #print(game_keys)
    '''
    for key in game_keys:
        if "P0" in key:
            print(key)
    '''
    
    white_key = state.player_key(game.state, Color.WHITE)
    red_key  = state.player_key(game.state, Color.RED)
    orange_key = state.player_key(game.state, Color.ORANGE)
    blue_key = state.player_key(game.state, Color.BLUE)
    player_keys = [white_key, red_key, orange_key, blue_key]
    #print(dict(filter(lambda item: white_key in item[0], game_keys.items())))
    max_points = 2
    max_player = None
    for p in player_keys:
        items = dict(filter(lambda item: p in item[0], game_keys.items()))
        player_points  = list(items.values())[0]
        #print(str(p) + ' ' + str(player_points))
        if player_points > max_points:
            max_points = player_points
            max_player = p
            
    print(str(max_player) + ' ' + str(max_points))
    return items
    
    


        
    #print(player_states)
    #print(game.state.player_state["P0_BRICK_IN_HAND"])
    return playable_actions[0]
    # ===== END YOUR CODE =====