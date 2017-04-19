"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""

from typing import Callable, Any, Tuple, List, Optional, Iterable, NamedTuple, Dict, Set
from numbers import Number

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

    # TODO: Strategy: Number of legal moves in two or three rounds assuming the opponent doesn't move

    own_moves = len(game.get_legal_moves(player))  # TODO: that should be the number of childs for the cached node
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)


class GraphEdge(NamedTuple):
    """An edge describing a move to get from one node to another."""
    top: 'GraphNode'
    bottom: 'GraphNode'
    move: Position


class GraphNodeCache:
    """Cache for already seen board states."""

    def __init__(self):
        self._registry = {}  # type: Dict[int, GraphNode]

    @staticmethod
    def _hash(board: Board):
        # Udacity's provided implementation hashes on strings generated from lists ...
        # I feel bad for directly accessing that field, but speed matters here and I
        # can't change the interface.
        return hash(tuple(board._board_state))

    def clear(self):
        """Clears the registry."""
        self._registry.clear()

    def find(self, branch: Board) -> Optional['GraphNode']:
        """Attempts to find a graph node given a board state.

        Parameters
        ----------
        branch : Board
            The branch node to explore.

        Returns
        -------
        GraphNode
            The cached node.
        None
            No cached node existed.
        """

        # TODO: It might be interesting to also check for symmetric states, e.g. rotations and transposes.
        key = self._hash(branch)
        return self._registry[key] if key in self._registry else None

    def register(self, node: 'GraphNode'):
        """Registers a new node with the cache.

        Parameters
        ----------
        node : GraphNode
            The node to register.
        """
        assert node.board is not None
        key = self._hash(node.board)
        assert key not in self._registry
        self._registry[key] = node

    def deregister(self, node: 'GraphNode'):
        """Deregisters an existing node from the cache.

        Parameters
        ----------
        node : GraphNode
            The graph node to remove.
        """
        assert node.board is not None
        key = self._hash(node.board)
        if key in self._registry:
            del self._registry[key]

    def __len__(self):
        return len(self._registry)


class GraphNode:
    """Graph node structure to maintain explored board states."""

    def __init__(self, registry: GraphNodeCache, branch: Optional[Board], score: float, age: int = 0):
        """
        Parameters
        ----------
        registry : GraphNodeCache
            A registry used for caching already explored board states and their corresponding node.       
        branch : Board
            The board state described by this node.
        score : float
            A utility or heuristics value judging the quality of the move.
        age : int
            The current age of the node.
        """
        self._in_edges = set()  # type: Set[GraphEdge]
        self._out_edges = []  # type: List[GraphEdge]
        self._moves = {}  # type: Dict[Position, GraphNode]
        self.registry = registry
        self.board = branch
        self.score = score
        self.age = age
        self.tainted = False
        self.expected_children = -1
        registry.register(self)

    def __del__(self):
        self.purge(self.age)

    def __str__(self):
        return 'Node at age {} - score {} - {} ancestors, {} descendants\n{}'\
            .format(self.age, self.score, len(self._in_edges), len(self._out_edges), self.board.to_string())

    def update_score(self, score: float):
        """Sets the score of this node and marks the ancestors as tainted.
        
        Parameters
        ----------
        score : float
            The new score.
        """
        if self.score == score:
            return
        self.score = score
        for edge in self._in_edges:
            edge.top.taint()

    def taint(self):
        """Marks this node as tainted, indicating that the outgoing edges need sorting."""
        self.tainted = True

    def set_age(self, age: int) -> int:
        """Sets the age of this node and its descendants.

        Parameters
        ----------
        age : int
            The new age.

        Returns
        -------
        int
            The previous age.
        """
        previous_age, self.age = self.age, age
        for edge in self._out_edges:
            edge.bottom.set_age(age)
        return previous_age

    @property
    def has_children(self) -> bool:
        return len(self._out_edges) > 0

    @property
    def has_seen_all_children(self) -> bool:
        """Determines if all children were discovered."""
        return len(self._out_edges) == self.expected_children

    def all_children_seen(self):
        """Marks that all children were discovered."""
        self.expected_children = len(self._out_edges)

    def children(self, should_sort: bool=True) -> Iterable[Tuple[Position, 'GraphNode']]:
        """Iterates the outgoing edges. 
        
        If this node is tainted, the outgoing edges will be sorted and the tainted flag will be reset.

        Returns
        -------
        Iterable[Tuple[Position, GraphNode]]
            A move leading to a node.
        """
        if self.tainted and should_sort:
            self.sort_children()
        for edge in self._out_edges:
            yield edge.move, edge.bottom

    def explore_child_boards(self, should_sort: bool=True) -> Iterable[Tuple[Position, Board]]:
        """Iterates known and unexplored outgoing edges.
        
        If this node is tainted, the outgoing edges will be sorted and the tainted flag will be reset.

        Returns
        -------
        Iterable[Tuple[bool, Position, GraphNode]]
            A move leading to a node.
        """
        if self.has_seen_all_children:
            return ((move, node.board) for move, node in self.children(should_sort))

        all_moves = set(self.board.get_legal_moves())
        known_moves = self._moves.keys()
        for move in all_moves:
            if move in known_moves:
                continue
            branch = self.board.forecast_move(move)
            self.add_child(move, branch, score=0.0)
            yield move, branch
        for move in (known_moves - all_moves):
            yield move, self._moves[move].board
        self.all_children_seen()

    def add_child(self, move: Position, branch: Board, score: float) -> 'GraphNode':
        """Adds a child to this node.

        Parameters
        ----------
        move : Position
            The move that resulted in the new child.
        branch : Board
            The board after the move has been made.
        score : float
            A utility or heuristics value judging the quality of the move.
            
        Returns
        -------
        GraphNode
            The child node.
        """
        child = self.registry.find(branch) or GraphNode(self.registry, branch, score, self.age)
        self._moves[move] = child
        edge = GraphEdge(top=self, bottom=child, move=move)
        child._in_edges.add(edge)
        self._out_edges.append(edge)
        self.taint()
        return child

    def sort_children(self):
        """Sorts the children by score in descending order."""
        self._out_edges = sorted(self._out_edges, key=lambda e: -e.bottom.score, reverse=False)
        self.tainted = False

    def make_root(self, new_age: int):
        """Makes this node the new root of the graph, purging its ancestors and all unrelated nodes.

        Parameters
        ----------
        new_age : int
            The new age of the node and its children. This allows for purging all
            nodes that are not descendants of this node.
        """
        # Painting our children with the new age masks them from purging.
        threshold_age = self.set_age(new_age)
        # Delete all ancestors, siblings and their children, but not OUR children.
        while len(self._in_edges) > 0:
            edge = self._in_edges.pop()  # type: GraphEdge
            edge.top.purge(threshold_age)

    def purge(self, threshold_age: int):
        """Removes this node from the graph, while also purging its ancestors and children.

        Parameters
        ---------
        threshold_age : int
            The age below which this node is purged. If the node is older than this number,
            it will not be purged and also not propagate the wave.
        """
        if self.age > threshold_age or self.board is None:
            return
        self.registry.deregister(self)
        self.board = None
        while len(self._in_edges) > 0:
            edge = self._in_edges.pop()  # type: GraphEdge
            edge.top.purge(threshold_age)
        while len(self._out_edges) > 0:
            edge = self._out_edges.pop()  # type: GraphEdge
            edge.bottom.purge(threshold_age)


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

        self.tree = None  # type: Optional[GraphNode]
        self.move_registry = GraphNodeCache()

        self.search = self.minimax if method == 'minimax' else self.alphabeta
        assert method == 'minimax' or method == 'alphabeta', \
            'The search method {} is not implemented.'.format(method)

    @property
    def is_unit_test(self):
        return self.score != custom_score

    def find_node(self, game: Board) -> Optional[GraphNode]:
        """Attempts to find the node belonging to the specified game state.
        
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
            
        Returns
        -------
        GraphNode
            The node.
        None
            No node was found.
        """
        return self.move_registry.find(game)

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

        self.tree = self.find_node(game) or \
                    GraphNode(self.move_registry, branch=game, score=0.0, age=game.move_count)
        self.tree.make_root(game.move_count)

        best_value, best_move = NEGATIVE_INFINITY, None
        depth = 0

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
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: Implement as queue

        # Fetch the current node. Normally, the node must not be None at this point,
        # but this assertion breaks the unit tests provided by Udacity.
        current_node = self.find_node(game)
        if current_node is None:
            self.move_registry.clear()
            current_node = GraphNode(self.move_registry, branch=game, score=float('nan'), age=game.move_count)

        # Termination criterion.
        if depth == 0 or game.is_winner(game.active_player) or game.is_loser(game.active_player):
            score = self.score(game, game.active_player if maximizing_player else game.inactive_player)
            current_node.update_score(score)
            return score, None

        # The infinities ensure that the first result always initializes the fields.
        best_value = NEGATIVE_INFINITY if maximizing_player else POSITIVE_INFINITY
        best_move = None

        # Explore the children
        for move, branch in current_node.explore_child_boards(should_sort=False):
            # TODO: Keep track of the explored depth along this branch. If in cache, don't explore if deep enough.
            v, m = self.minimax(branch, depth - 1, maximizing_player=not maximizing_player)
            if maximizing_player:
                if v > best_value:
                    best_value, best_move = v, move
            else:
                if v < best_value:
                    best_value, best_move = v, move

        current_node.all_children_seen()
        current_node.update_score(best_value)
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

        # Fetch the current node. Normally, the node must not be None at this point,
        # but this assertion breaks the unit tests provided by Udacity.
        current_node = self.find_node(game)
        if current_node is None:
            self.move_registry.clear()
            current_node = GraphNode(self.move_registry, branch=game, score=float('nan'), age=game.move_count)

        # Termination criterion.
        if depth == 0 or game.is_winner(game.active_player) or game.is_loser(game.active_player):
            score = self.score(game, game.active_player if maximizing_player else game.inactive_player)
            current_node.update_score(score)
            return score, None

        # The infinities ensure that the first result always initializes the fields.
        best_value = NEGATIVE_INFINITY if maximizing_player else POSITIVE_INFINITY
        best_move = None

        # TODO: Maybe move branch to aid prediction
        for move, branch in current_node.explore_child_boards(should_sort=True):
            v, m = self.alphabeta(branch, depth-1, alpha=alpha, beta=beta, maximizing_player=not maximizing_player)
            if maximizing_player:
                if v > best_value:
                    best_value, best_move = v, move
                alpha = max(alpha, v)  # raise the lower bound
                if v >= beta:  # TODO: add explanatory comment
                    break
            else:
                if v < best_value:
                    best_value, best_move = v, move
                beta = min(beta, v)  # lower the upper bound
                if v <= alpha:  # TODO: add explanatory comment
                    break

        current_node.all_children_seen()
        current_node.update_score(best_value)
        return best_value, best_move
