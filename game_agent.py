"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""

from typing import Callable, Any, Tuple, List, Optional, Iterable
from numbers import Number
from operator import itemgetter

from isolation import Board


Score = float
Position = Tuple[int, int]
CandidateMove = Tuple[Score, Optional[Position]]
TimerFunction = Callable[[], Number]

NEGATIVE_INFINITY = float("-inf")
POSITIVE_INFINITY = float("inf")


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game: Board, player: Any) -> float:
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

    # TODO: finish this function!
    raise NotImplementedError


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)  This parameter should be ignored when iterative = True.

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).  When True, search_depth should
        be ignored and no limit to search depth.

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth: int=3, score_fn: Callable[[Board, Any], float]=custom_score,
                 iterative: bool=True, method: str='minimax', timeout: float=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn  # type: Callable[[Board, Any], float]
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.method = method

        self.search = self.minimax if method == 'minimax' else self.alphabeta
        assert method == 'minimax' or method == 'alphabeta', \
            'The search method {} is not implemented.'.format(method)

    def get_move(self, game: Board, legal_moves: List[Position], time_left: TimerFunction) -> Position:
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            DEPRECATED -- This argument will be removed in the next release.
            Corresponds to the result of board.get_legal_moves().

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

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        # TODO: Initializations, opening moves, etc.
        # TODO: Depending on the choice of the opponent, we can remove parts of the previously built tree and clean dictionaries (hash collision -> linear search)

        position = None
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            _, position = self.search(game, depth=self.search_depth, maximizing_player=True)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        assert position is not None
        return position

    def terminal_score(self, game: Board, depth: int) -> Optional[float]:
        """
        Decides if the game is at a terminal state and returns the score or heuristic, otherwise returns None.
        
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
            The heuristic value or utility if this is a terminal state.
        
        None
            Indicates this is not a terminal state.
        """
        player = game.active_player
        if game.is_winner(player) or game.is_loser(player):
            return game.utility(game.active_player)
        elif depth == 0:
            return self.score(game, player)
        return None

    @staticmethod
    def move_branches(game: Board) -> Iterable[Tuple[Position, Board]]:
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
        for m in game.get_legal_moves():
            yield m, game.forecast_move(m)

    def minimax(self, game: Board, depth: int, maximizing_player: bool=True) -> CandidateMove:
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        assert self.time_left is not None
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: Implement as queue

        best_score = NEGATIVE_INFINITY
        best_move = None
        for move, branch in self.move_branches(game):
            v = self.minimax_min(game, depth-1)
            if v > best_score:
                best_score = v
                best_move = move

        return best_score, best_move if best_move is not None else (-1, -1)

    def minimax_max(self, game: Board, depth: int) -> Score:
        """
        Performs the max player step of minimax. 

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
            The utility value or its best heuristic.
        """
        v = self.terminal_score(game, depth)
        if v is not None:
            return v
        v = NEGATIVE_INFINITY
        for _, branch in self.move_branches(game):
            # TODO: Add caching/lookup - check for symmetries
            # TODO: Rotation/Mirroring of a cache hit is not required until that branch is "physically" taken
            v = max(v, self.minimax_min(branch, depth=depth-1))
        return v

    def minimax_min(self, game: Board, depth: int) -> Score:
        """
        Performs the min player step of minimax. 

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
            The utility value or its best heuristic.
        """
        v = self.terminal_score(game, depth)
        if v is not None:
            return v
        v = POSITIVE_INFINITY
        for _, branch in self.move_branches(game):
            # TODO: Add caching/lookup - check for symmetries
            # TODO: Rotation/Mirroring of a cache hit is not required until that branch is "physically" taken
            v = min(v, self.minimax_max(branch, depth=depth-1))
        return v

    def alphabeta(self, game: Board, depth: int, alpha: float=float("-inf"), beta: float=float("inf"),
                  maximizing_player: bool=True) -> CandidateMove:
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

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

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        assert self.time_left is not None
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: Sort for efficient pruning.
        # TODO: Implement as queue

        player = game.active_player
        if depth == 0 or game.is_winner(player) or game.is_loser(player):
            return self.score(game, player), None

        # The infinities ensure that the first result always initializes the fields.
        best_value = NEGATIVE_INFINITY if maximizing_player else POSITIVE_INFINITY
        best_move = None

        for move, branch in self.move_branches(game):
            v, m = self.alphabeta(branch, depth-1, alpha=alpha, beta=beta, maximizing_player=not maximizing_player)
            if maximizing_player:
                if v > best_value:
                    best_value, best_move = v, m
                alpha = max(alpha, v)  # raise the lower bound
                if v >= beta:
                    break
            else:
                if v < best_value:
                    best_value, best_move = v, m
                beta = min(beta, v)  # lower the upper bound
                if v <= alpha:
                    break
        return best_value, best_move
