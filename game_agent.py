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

    This is the "improved" evaluation function discussed in lecture that outputs a
    score equal to the difference in the number of moves available to the
    two players.

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

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)


class TreeNode:
    _registry = {}

    @staticmethod
    def find_node(branch: Board) -> Optional['TreeNode']:
        return TreeNode._registry[branch] if branch in TreeNode._registry else None

    def __init__(self, parent: Optional['TreeNode'], move: Optional[Position], branch: Board):
        self._parent = parent
        self._children = []
        self._branch = branch
        self._move = move
        TreeNode._registry[branch] = self

    def __del__(self):
        for child in self.children:
            del child
        if self.parent is not None:
            del self._parent
        del TreeNode._registry[self.branch]
        self._branch = None
        self._move = None

    def __hash__(self):
        return self._branch.hash()

    def add_childs(self, children: Iterable['TreeNode']):
        for child in children:
            self._children.append(child)

    def forget_child(self, child: 'TreeNode'):
        self._children.remove(child)

    @property
    def parent(self) -> Optional['TreeNode']:
        return self._parent

    @property
    def children(self) -> Iterable['TreeNode']:
        return self._children

    @property
    def has_children(self) -> bool:
        return any(self._children)

    @property
    def siblings(self) -> Iterable['TreeNode']:
        if self._parent is None:
            return []
        return (child for child in self._parent.children if child is not self)

    @property
    def branch(self) -> Optional[Board]:
        return self._branch

    @property
    def move(self) -> Optional[Position]:
        return self._move

    def make_root(self):
        if self.parent is not None:
            self.parent.forget_child(self)
        for sibling in self.siblings:
            del sibling


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
        self.tree = None  # type: Optional[TreeNode]

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

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        # TODO: Initializations, opening moves, etc.
        # TODO: Depending on the choice of the opponent, we can remove parts of the previously built tree and clean dictionaries (hash collision -> linear search). Can we use the game round?

        best_value, best_move = NEGATIVE_INFINITY, None
        depth = 0

        if self.tree is None:
            self.tree = TreeNode(None, None, game)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            if self.iterative:
                while True:
                    depth += 1
                    v, m = self.search(game, depth=depth, maximizing_player=True)
                    if v > best_value:
                        best_value, best_move = v, m
            else:
                depth = self.search_depth
                best_value, best_move = self.search(game, depth=depth, maximizing_player=True)

        except Timeout:
            # TODO: Handle any actions required at timeout, if necessary
            # print('Reached depth {} in move {}'.format(depth, game.move_count))
            pass

        # Return the best move from the last completed search iteration
        return best_move

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
        for m in game.get_legal_moves(game.active_player):
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

        player = game.active_player
        if depth == 0 or game.is_winner(player) or game.is_loser(player):
            return self.score(game, game.active_player if maximizing_player else game.inactive_player), None

        # The infinities ensure that the first result always initializes the fields.
        best_value = NEGATIVE_INFINITY if maximizing_player else POSITIVE_INFINITY
        best_move = None

        root = TreeNode.find_node(game) or TreeNode(None, None, game)
        if not root.has_children:
            root.add_childs((TreeNode(root, move, game.forecast_move(move)) for move in game.get_legal_moves()))

        # TODO: Move is not a property of a child, but of an edge TO the child. Otherwise different parents couldn't refer to the same game state.
        # TODO: Beware circular cleanups if a node reuses a different node's parent. Maybe track all parents and purge DOWN only if all parents are gone?

        for node in root.children:
            move, branch = node.move, node.branch
            v, m = self.minimax(branch, depth - 1, maximizing_player=not maximizing_player)
            if maximizing_player:
                if v > best_value:
                    best_value, best_move = v, move
            else:
                if v < best_value:
                    best_value, best_move = v, move
        return best_value, best_move

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

        # TODO: Sort for efficient pruning. This requires knowledge of the previous expansions, because search is depth-first, not breadth-first.
        # TODO: Implement as queue

        player = game.active_player
        if depth == 0 or game.is_winner(player) or game.is_loser(player):
            return self.score(game, game.active_player if maximizing_player else game.inactive_player), None

        # The infinities ensure that the first result always initializes the fields.
        best_value = NEGATIVE_INFINITY if maximizing_player else POSITIVE_INFINITY
        best_move = None

        # TODO: Maybe move branch to aid prediction
        for move, branch in self.move_branches(game):
            v, m = self.alphabeta(branch, depth-1, alpha=alpha, beta=beta, maximizing_player=not maximizing_player)
            if maximizing_player:
                # If the value is better, store it and the move that led to it.
                if v > best_value:
                    best_value, best_move = v, move
                alpha = max(alpha, v)  # raise the lower bound
                if v >= beta:  # TODO: add explanatory comment
                    break
            else:
                # If the value is better, store it and the move that led to it.
                if v < best_value:
                    best_value, best_move = v, move
                beta = min(beta, v)  # lower the upper bound
                if v <= alpha:  # TODO: add explanatory comment
                    break
        return best_value, best_move
