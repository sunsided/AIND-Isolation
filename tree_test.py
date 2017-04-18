import unittest

import isolation
from game_agent import GraphNode, GraphEdge, GraphNodeCache


class TreeTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TreeTest, self).__init__(*args, **kwargs)
        self.registry = GraphNodeCache()

    def test_creating_nodes(self):
        """Test output interface of heuristic score function interface."""
        game = isolation.Board("Player1", "Player2")

        #             a
        #             |
        #             b
        #             |
        #             c

        a = GraphNode(self.registry, game, 0.0)
        b = a.add_child((0, 0), a.board.forecast_move((0, 0)), 1.0)
        b.add_child((1, 1), b.board.forecast_move((1, 1)), 1.0)

        self.assertEqual(len(self.registry), 3,
                         "There should be three nodes in the registry")

    def test_creating_nodes_diamond(self):
        """Test output interface of heuristic score function interface."""
        game = isolation.Board("Player1", "Player2")

        #                 a        (root)
        #               /   \
        #             a1     c1     max
        #              |     |
        #             b1     d1     min
        #              |     |
        #             a2     c2     max
        #              |     |
        #             b2     d2     min
        #               \   /
        #               a3,c3       max

        a = GraphNode(self.registry, game, 0.0)
        a1 = a.add_child((2, 0), a.board.forecast_move((2, 0)), 1.0)
        b1 = a1.add_child((1, 0), a1.board.forecast_move((1, 0)), 1.0)
        a2 = b1.add_child((2, 1), b1.board.forecast_move((2, 1)), 1.0)
        b2 = a2.add_child((1, 1), a2.board.forecast_move((1, 1)), 1.0)
        a3 = b2.add_child((3, 3), b2.board.forecast_move((3, 3)), 1.0)

        c1 = a.add_child((2, 1), a.board.forecast_move((2, 1)), 1.0)
        d1 = c1.add_child((1, 0), c1.board.forecast_move((1, 0)), 1.0)
        c2 = d1.add_child((2, 0), d1.board.forecast_move((2, 0)), 1.0)
        d2 = c2.add_child((1, 1), c2.board.forecast_move((1, 1)), 1.0)
        c3 = d2.add_child((3, 3), d2.board.forecast_move((3, 3)), 1.0)

        self.assertEqual(len(self.registry), 10,
                         "There should be four nodes in the registry")

    def test_deleting_tree_from_root(self):
        """Test output interface of heuristic score function interface."""
        game = isolation.Board("Player1", "Player2")

        #               a
        #           /   |   \
        #         b1   b2   b3
        #          |         |
        #         c1         c3
        #          |
        #         d1

        a = GraphNode(self.registry, game, 0.0)
        b1 = a.add_child((0, 0), a.board.forecast_move((0, 0)), 1.0)
        b2 = a.add_child((0, 1), a.board.forecast_move((0, 1)), 1.0)
        a.add_child((0, 2), a.board.forecast_move((0, 2)), 1.0)
        c1 = b1.add_child((1, 1), b1.board.forecast_move((1, 1)), 1.0)
        b2.add_child((1, 2), b2.board.forecast_move((1, 2)), 1.0)
        c1.add_child((2, 0), c1.board.forecast_move((2, 0)), 1.0)

        a.purge(a.age)

        self.assertEqual(len(self.registry), 0,
                         "There should be no remaining nodes in the registry")

    def test_deleting_tree_from_node(self):
        """Test output interface of heuristic score function interface."""
        game = isolation.Board("Player1", "Player2")

        #               a
        #           /   |   \
        #         b1   b2   b3
        #          |         |
        #         c1         c3
        #          |
        #         d1

        a = GraphNode(self.registry, game, 0.0)
        b1 = a.add_child((0, 0), a.board.forecast_move((0, 0)), 1.0)
        b2 = a.add_child((0, 1), a.board.forecast_move((0, 1)), 1.0)
        a.add_child((0, 2), a.board.forecast_move((0, 2)), 1.0)
        c1 = b1.add_child((1, 1), b1.board.forecast_move((1, 1)), 1.0)
        b2.add_child((1, 2), b2.board.forecast_move((1, 2)), 1.0)
        c1.add_child((2, 0), c1.board.forecast_move((2, 0)), 1.0)

        b2.purge(b2.age)

        self.assertEqual(len(self.registry), 0,
                         "There should be no remaining nodes in the registry")

    def test_child_sorting(self):
        """Test output interface of heuristic score function interface."""
        game = isolation.Board("Player1", "Player2")

        #               a
        #           /   |   \
        #         b1   b2   b3
        #          |         |
        #         c1         c3
        #          |
        #         d1

        a = GraphNode(self.registry, game, 0.0)
        b1 = a.add_child((0, 0), a.board.forecast_move((0, 0)), 2.0)
        b2 = a.add_child((0, 1), a.board.forecast_move((0, 1)), 3.0)
        a.add_child((0, 2), a.board.forecast_move((0, 2)), 4.0)
        c1 = b1.add_child((1, 1), b1.board.forecast_move((1, 1)), 1.0)
        b2.add_child((1, 2), b2.board.forecast_move((1, 2)), 1.0)
        c1.add_child((2, 0), c1.board.forecast_move((2, 0)), 1.0)

        score = float('inf')
        for child in a.children:
            self.assertLess(child.bottom.score, score,
                            "Children must be sorted by score in descending order")
            score = child.bottom.score

    def test_make_root(self):
        """Test output interface of heuristic score function interface."""
        game = isolation.Board("Player1", "Player2")

        #               a
        #           /   |   \
        #         b1   b2   b3
        #          |         |
        #         c1         c3
        #          |
        #         d1

        a = GraphNode(self.registry, game, 0.0)
        b1 = a.add_child((0, 0), a.board.forecast_move((0, 0)), 1.0)
        b2 = a.add_child((0, 1), a.board.forecast_move((0, 1)), 1.0)
        a.add_child((0, 2), a.board.forecast_move((0, 2)), 1.0)
        c1 = b1.add_child((1, 1), b1.board.forecast_move((1, 1)), 1.0)
        b2.add_child((1, 2), b2.board.forecast_move((1, 2)), 1.0)
        c1.add_child((2, 0), c1.board.forecast_move((2, 0)), 1.0)

        c1.make_root(c1.age + 1)

        self.assertEqual(len(self.registry), 2,
                         "There should be no only c1 and its childs in the registry")

    def test_make_root_diamond(self):
        """Test output interface of heuristic score function interface."""
        game = isolation.Board("Player1", "Player2")

        #                 a        (root)
        #               /   \
        #             a1     c1     max
        #              |     |
        #             b1     d1     min
        #              |     |
        #             a2     c2     max
        #              |     |
        #             b2     d2     min
        #               \   /
        #               a3,c3       max

        a = GraphNode(self.registry, game, 0.0)
        a1 = a.add_child((2, 0), a.board.forecast_move((2, 0)), 1.0)
        b1 = a1.add_child((1, 0), a1.board.forecast_move((1, 0)), 1.0)
        a2 = b1.add_child((2, 1), b1.board.forecast_move((2, 1)), 1.0)
        b2 = a2.add_child((1, 1), a2.board.forecast_move((1, 1)), 1.0)
        a3 = b2.add_child((3, 3), b2.board.forecast_move((3, 3)), 1.0)

        c1 = a.add_child((2, 1), a.board.forecast_move((2, 1)), 1.0)
        d1 = c1.add_child((1, 0), c1.board.forecast_move((1, 0)), 1.0)
        c2 = d1.add_child((2, 0), d1.board.forecast_move((2, 0)), 1.0)
        d2 = c2.add_child((1, 1), c2.board.forecast_move((1, 1)), 1.0)
        c3 = d2.add_child((3, 3), d2.board.forecast_move((3, 3)), 1.0)

        a2.make_root(a2.age + 1)

        self.assertEqual(len(self.registry), 3,
                         "Only the node a2 and its descendants should be kept")
