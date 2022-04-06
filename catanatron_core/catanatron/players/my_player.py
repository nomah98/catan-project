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
    key = state.player_key(game.state, Color.WHITE)
    red_index = game.state.color_to_index[Color.RED]
    # print(list(map(game_keys.get, filter(lambda x:x in "P0", game_keys))))
    #print(dict(filter(lambda item: key in item[0], game_keys.items())))


        
    #print(player_states)
    #print(game.state.player_state["P0_BRICK_IN_HAND"])
    return playable_actions[0]
    # ===== END YOUR CODE =====