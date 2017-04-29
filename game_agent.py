"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

NEGATIVE_INFINITY = float("-inf")
POSITIVE_INFINITY = float("inf")


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    own_moves = len(game.get_legal_moves(player))
    if own_moves == 0:
        return NEGATIVE_INFINITY

    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    if opp_moves == 0:
        return POSITIVE_INFINITY

    opp_future_moves = len(__get_moves_2(game, game.get_opponent(player)))
    return float(own_moves - 2*opp_future_moves)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    own_moves = len(game.get_legal_moves(player))
    if own_moves == 0:
        return NEGATIVE_INFINITY

    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    if opp_moves == 0:
        return POSITIVE_INFINITY

    p1 = game.get_player_location(game.active_player)
    p2 = game.get_player_location(game.get_opponent(player))
    dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    return float(own_moves - 2 * opp_moves - dist)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    own_moves = len(game.get_legal_moves(player))
    if own_moves == 0:
        return NEGATIVE_INFINITY

    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    if opp_moves == 0:
        return POSITIVE_INFINITY

    p1 = game.get_player_location(game.active_player)
    p2 = game.get_player_location(game.get_opponent(player))
    dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    opp_future_moves = len(__get_moves_2(game, game.get_opponent(player)))
    return float(own_moves - 2 * opp_future_moves - dist)


def __get_moves_2(game, player):
    """Generate the list of possible moves for an L-shaped motion (like a
    knight in chess).
    """

    loc = game.get_player_location(player)

    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
    directions = set(((r + dr, c + dc) for dr, dc in directions for r, c in directions))

    r, c = loc
    valid_moves = [(r + dr, c + dc) for dr, dc in directions
                   if game.move_is_legal((r + dr, c + dc))]
    random.shuffle(valid_moves)
    return set(valid_moves)


def move_branches(game):
    """
    Determines the legal moves in the current state, creates branches and yields them.

    Parameters
    ----------
    game : isolation.Board
        An instance of the Isolation game `Board` class representing the
        current game state

    Returns
    -------
    Iterable[Position, Board]
        The moves and their respective branch of the board.
    """
    for m in game.get_legal_moves(game.active_player):
        yield m, game.forecast_move(m)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration.
        # Note that this only applies to iterative deepening approaches.
        return -1, -1

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Note that this step is technically identical to the _minimax_max()
        # function, but additionally keeps track of the move to take eventually.
        best_value, best_move = max(((self._minimax_min(branch, depth - 1), move)
                                     for move, branch in move_branches(game)),
                                    key=lambda t: t[0])
        return best_move or (-1, -1)

    def _minimax_min(self, game, depth):
        """Implements a depth-limited minimax search step for the min player.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        float
            The lowest score (heuristic) possible in this state.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        player = game.active_player
        if depth == 0 or game.is_winner(player) or game.is_loser(player):
            return self.score(game, game.get_opponent(player))

        return min(self._minimax_max(branch, depth - 1)
                   for move, branch in move_branches(game))

    def _minimax_max(self, game, depth):
        """Implements a depth-limited minimax search step for the max player.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        float
            The highest score (heuristic) possible in this state.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        player = game.active_player
        if depth == 0 or game.is_winner(player) or game.is_loser(player):
            return self.score(game, player)

        return max(self._minimax_min(branch, depth - 1)
                   for move, branch in move_branches(game))


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        best_move = None
        try:
            depth = 0
            while True:
                depth += 1
                best_move = self.alphabeta(game, depth=depth)
        except SearchTimeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return best_move or (-1, -1)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_value, best_move = self._alphabeta_max(game, depth , alpha=alpha, beta=beta)
        return best_move

    def _alphabeta_min(self, game, depth, alpha, beta):
        """Implements a depth-limited alpha-beta pruned minimax search step for the min player.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
            
        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        float
            The lowest score (heuristic) possible in this state.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        player = game.active_player
        if depth == 0 or game.is_winner(player) or game.is_loser(player):
            return self.score(game, game.get_opponent(player)), None

        best_value, best_move = None, None
        for move, branch in move_branches(game):
            v, m = self._alphabeta_max(branch, depth - 1, alpha=alpha, beta=beta)
            if best_value is None or v < best_value:
                best_value, best_move = v, move

            # If the value is lower than the current lower bound available to
            # the calling max player, we know that this branch will never be taken
            # (since max will at least select the known branch that created the
            # higher lower bound). Because of that, we can stop searching.
            if v <= alpha:
                break

            # lower the upper bound
            beta = min(beta, v)

        return best_value, best_move

    def _alphabeta_max(self, game, depth, alpha, beta):
        """Implements a depth-limited alpha-beta pruned minimax search step for the max player.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
            
        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        float
            The highest score (heuristic) possible in this state.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        player = game.active_player
        if depth == 0 or game.is_winner(player) or game.is_loser(player):
            return self.score(game, player), None

        best_value, best_move = None, None
        for move, branch in move_branches(game):
            v, m = self._alphabeta_min(branch, depth - 1, alpha=alpha, beta=beta)
            if best_value is None or v > best_value:
                best_value, best_move = v, move

            # If the value is higher than the current upper bound available to
            # the calling min player, we know that this branch will never be taken
            # (since min will at least select the known branch that created the
            # lower upper bound). Because of that, we can stop searching.
            if v >= beta:
                break

            # raise the lower bound
            alpha = max(alpha, v)

        return best_value, best_move
