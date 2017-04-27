"""Implement your own custom search agent using any combination of techniques
you choose.  This agent will compete against other students (and past
champions) in a tournament.

         COMPLETING AND SUBMITTING A COMPETITION AGENT IS OPTIONAL
"""
from typing import Callable, Any, Tuple, List, Optional, Iterable, NamedTuple, Dict, Set
from numbers import Number
from random import random
from math import isinf

from isolation import Board


Score = float
Position = Tuple[int, int]
CandidateMove = Tuple[Score, Optional[Position]]
TimerFunction = Callable[[], Number]

NEGATIVE_INFINITY = float("-inf")
POSITIVE_INFINITY = float("inf")

DEBUG = False


def log(message: str):
    """Prints logging information if debugging is enabled.

    Parameters
    ----------
    message : str
        The message to print.
    """
    if DEBUG:
        print(message)


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

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
    # TODO: Strategy: Number of legal moves in two or three rounds assuming the opponent doesn't move

    own_moves = len(game.get_legal_moves(player))  # TODO: that should be the number of childs for the cached node
    if own_moves == 0:
        return NEGATIVE_INFINITY

    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    if opp_moves == 0:
        return POSITIVE_INFINITY

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
        # return self._registry[key] if key in self._registry else None
        try:
            return self._registry[key]
        except KeyError:
            return None

    def register(self, node: 'GraphNode'):
        """Registers a new node with the cache.

        Parameters
        ----------
        node : GraphNode
            The node to register.
        """
        assert node.board is not None
        key = self._hash(node.board)
        # assert key not in self._registry -- assumes no custom behavior for unit tests
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
    node_counter = 0

    def __init__(self, registry: GraphNodeCache, branch: Optional[Board], score: float,
                 age: int = 0, depth: int = 0):
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
        self.registry = registry
        self.board = branch
        self.score = score
        self.age = age
        self.depth = depth
        self.id = GraphNode.node_counter
        self._in_edges = set()  # type: Set[GraphEdge]
        self._out_edges = []  # type: List[GraphEdge]
        self._moves = {}  # type: Dict[Position, GraphNode]
        self._has_seen_all_children = False
        self._children_need_sorting = False

        GraphNode.node_counter += 1
        registry.register(self)

    def __del__(self):
        self.purge(self.age)

    def __str__(self):
        return 'Depth {}, age {}, score {}, {} ancestors, {} descendants' \
            .format(self.depth, self.age, self.score, len(self._in_edges), len(self._out_edges))

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
        self._children_need_sorting = True

    def set_age_and_depth(self, age: int, depth: int) -> int:
        """Updates the age and depth of this node and its descendants. This is triggered by a tree root change.

        We have to keep in mind that this is a potentially cyclic graph, so deep nodes may point to a node
        that was already discovered further up the stream on a different branch.

        Parameters
        ----------
        age : int
            The new age.
        depth : int
            The new depth to set for this node. Descendants will a depth that is higher by 1.

        Returns
        -------
        int
            The previous age.
        """
        previous_age, self.age = self.age, age
        self.depth = depth
        for edge in self._out_edges:
            edge.bottom.set_age_and_depth(age, depth + 1)
        return previous_age

    @property
    def has_children(self) -> bool:
        """Determines if this node has children.

        Returns
        -------
        bool
            True if this node has children; False otherwise.
        """
        return len(self._out_edges) > 0

    @property
    def has_seen_all_children(self) -> bool:
        """Determines if all children were discovered.

        Returns
        -------
        bool
            True if all children of this node have been explored (at this depth); False if
            there are (possibly) unseen children.
        """
        return self._has_seen_all_children

    def all_children_seen(self):
        """Marks that all children were discovered."""
        self._has_seen_all_children = True

    def children(self, should_sort: bool = True) -> Iterable[Tuple[Position, 'GraphNode']]:
        """Iterates the outgoing edges. 

        If this node is tainted, the outgoing edges will be sorted and the tainted flag will be reset.

        Returns
        -------
        Iterable[Tuple[Position, GraphNode]]
            A move leading to a node.
        """
        if self._children_need_sorting and should_sort:
            self.sort_children()
        for edge in self._out_edges:
            yield edge.move, edge.bottom

    def explore_child_boards(self) -> Iterable[Tuple[Position, Board]]:
        """Iterates known and unexplored outgoing edges.

        If this node is tainted, the outgoing edges will be sorted and the tainted flag will be reset.
        The behavior in this method can be optimized if it is known that search will always
        use iterative deepening, which is not true given the unit tests.

        Returns
        -------
        Iterable[Tuple[bool, Position, GraphNode]]
            A move leading to a node.
        """
        if self.has_seen_all_children:
            for move, node in self.children():
                yield move, node.board

        all_moves = set(self.board.get_legal_moves())
        known_moves = set(self._moves.keys())
        remaining_moves = set(known_moves - all_moves)

        # First, we'll explore all potentially good and potentially winning moves.
        for move, node in self.children():
            if node.score < 0:
                continue
            yield move, node.board
            if move in remaining_moves:
                remaining_moves.remove(move)

        # We then iteratively explore all moves that we have not yet seen,
        # as they might contain vital information we don't know about.
        for move in all_moves:
            if move in known_moves:
                continue
            branch = self.board.forecast_move(move)
            self.add_child(move, branch)
            yield move, branch

        self.all_children_seen()

        # After that, we explore all remaining moves we already know from an
        # earlier iteration, while keeping the previously established sorting order.
        # This should result in exploring potentially bad moves last.
        for move, node in self.children():
            if move not in remaining_moves:
                continue
            yield move, node.board

    def add_child(self, move: Position, branch: Board, score: float = random() - .5) -> 'GraphNode':
        """Adds a child to this node.

        Parameters
        ----------
        move : Position
            The move that resulted in the new child.
        branch : Board
            The board after the move has been made.
        score : float
            A utility or heuristics value judging the quality of the move. If unset,
            the score is initialized to a small random number, increasing it's
            chance to be explored before already known, possible tie moves (score 0).

        Returns
        -------
        GraphNode
            The child node.
        """
        child = self.registry.find(branch) or \
                GraphNode(self.registry, branch, score, self.age, self.depth + 1)
        self._moves[move] = child
        edge = GraphEdge(top=self, bottom=child, move=move)
        child._in_edges.add(edge)
        self._out_edges.append(edge)
        self.taint()
        return child

    def sort_children(self):
        """Sorts the children by score in descending order."""
        self._out_edges = sorted(self._out_edges, key=lambda e: -e.bottom.score, reverse=False)
        self._children_need_sorting = False

    def make_root(self, new_age: int):
        """Makes this node the new root of the graph, purging its ancestors and all unrelated nodes.

        Parameters
        ----------
        new_age : int
            The new age of the node and its children. This allows for purging all
            nodes that are not descendants of this node.
        """
        # Shifting the tree depth, making this node depth zero; also
        # painting our children with the new age masks them from purging.
        threshold_age = self.set_age_and_depth(new_age, 0)
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
    """Game-playing agent to use in the optional player vs player Isolation
    competition.

    You must at least implement the get_move() method and a search function
    to complete this class, but you may use any of the techniques discussed
    in lecture or elsewhere on the web -- opening books, MCTS, etc.

    **************************************************************************
          THIS CLASS IS OPTIONAL -- IT IS ONLY USED IN THE ISOLATION PvP
        COMPETITION.  IT IS NOT REQUIRED FOR THE ISOLATION PROJECT REVIEW.
    **************************************************************************

    Parameters
    ----------
    data : string
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted.  Note that
        the PvP competition uses more accurate timers that are not cross-
        platform compatible, so a limit of 1ms (vs 10ms for the other classes)
        is generally sufficient.
    """

    def __init__(self, data=None, timeout=1.):
        self.score = custom_score
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.tree = None  # type: Optional[GraphNode]
        self.move_registry = GraphNodeCache()

    @property
    def is_unit_test(self) -> bool:
        """Determines if the code is running under a unit test.

        Returns
        -------
        bool
            True if this test is assumed to run in a unit test; False otherwise.
        """
        if self.score == custom_score:
            return False
        name = str(self.score)
        return 'test' in name or 'Eval' in name

    def find_node(self, game: Board) -> Optional[GraphNode]:
        """Attempts to find the node belonging to the specified game state.

        This function always returns None if `is_unit_test` returns True.
        This is because the test_get_move unit test expects the implementation
        to search all the nodes explicitly, regardless if the board state was
        already seen in a different tree or not.

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
        return self.move_registry.find(game) if not self.is_unit_test else None

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

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

        self.tree = self.find_node(game)
        if self.tree is None:
            self.tree = GraphNode(self.move_registry, branch=game, score=0.0, age=game.move_count, depth=0)
        else:
            self.tree.make_root(game.move_count)

        best_move = None
        depth = 0

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            # TODO: Determine the deepest fully explored depth of the Graph and start ID with this depth.

            log('Starting search ...')
            while self.time_left() > self.TIMER_THRESHOLD:
                depth += 1
                log('Beginning iterative deepening with depth {}'.format(depth))
                best_move = self.alphabeta(game, depth=depth, maximizing_player=True)

        except SearchTimeout:
            # TODO: Handle any actions required at timeout, if necessary
            pass
        finally:
            log('Timeout. Reached depth {} in move {}'.format(depth, game.move_count))

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game: Board, depth: int, alpha: float = float("-inf"), beta: float = float("inf"),
                  maximizing_player: bool = True) -> CandidateMove:
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
            raise SearchTimeout()

        # TODO: Implement as queue

        # Fetch the current node.
        current_node = self.get_current_node(game, depth)

        # Cached termination criterion.
        # TODO: for some reason, this doesn't work.
        # Checking for isinf(s) works only when the opponent plays optimally.
        # if current_node.score == POSITIVE_INFINITY:
        #     assert any(s.bottom.score == POSITIVE_INFINITY for s in current_node._out_edges) or len(current_node._out_edges) == 0
        #     return current_node.score, None

        # Termination criterion.
        if depth == 0 or game.is_winner(game.active_player) or game.is_loser(game.active_player):
            score = self.score(game, game.active_player if maximizing_player else game.inactive_player)
            current_node.update_score(score)
            return score, None

        # Explore the children.
        best_value, best_move = current_node.score, None
        try:
            for move, branch in current_node.explore_child_boards():
                v, m = self.alphabeta(branch, depth - 1, alpha=alpha, beta=beta,
                                      maximizing_player=not maximizing_player)
                if best_move is None:
                    best_value, best_move = v, move
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
        finally:
            if best_move is not None:
                current_node.update_score(best_value)

        return best_value, best_move

    def get_current_node(self, game: Board, depth: int) -> GraphNode:
        """Obtains the current graph node.

        Parameters
        ----------
        game : Board
            The current game state.
        depth : int
            The current depth.

        Returns
        -------
        GraphNode
            The current graph node.
        """
        # Normally, the node must not be None at this point, but
        # there are two ways this might fail:
        # 1) The unit tests provided by Udacity call this method directly, so
        #    the initialization from the get_move method is missing.
        # 2) The opponent could have taken a move we did not explore yet.
        current_node = self.find_node(game)
        if current_node is None:
            assert self.is_unit_test, 'This assumption should only hold in unit tests but failed in move {}, depth {} ({}).'.format(
                game.move_count, depth, self.score)
            self.move_registry.clear()
            current_node = GraphNode(self.move_registry, branch=game, score=0, age=game.move_count)

        return current_node
