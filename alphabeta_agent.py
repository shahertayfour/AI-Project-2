from agent_base import Agent
from game import GameState
from evaluation import betterEvaluationFunction

class AlphaBetaAgent(Agent):
    def get_action(self, state: GameState, depth=None):
        value, action = self.alphabeta(state, alpha=float('-inf'), beta=float('inf'),
                                       depth_limit=depth, current_depth=0)
        if action is None:
            legal = state.get_legal_actions()
            if legal:
                action = legal[0]
        return action

    def alphabeta(self, state: GameState, alpha, beta, depth_limit, current_depth):
        if state.is_terminal():
            return state.utility(), None

        if depth_limit is not None and current_depth >= depth_limit:
            return betterEvaluationFunction(state), None

        legal_actions = state.get_legal_actions()

        if state.to_move == 'X':
            max_value = float('-inf')
            best_action = None

            for action in legal_actions:
                successor = state.generate_successor(action)
                value, _ = self.alphabeta(successor, alpha, beta, depth_limit, current_depth + 1)

                if value > max_value:
                    max_value = value
                    best_action = action

                alpha = max(alpha, value)

                if alpha >= beta:
                    break

            return max_value, best_action

        else:
            min_value = float('inf')
            best_action = None

            for action in legal_actions:
                successor = state.generate_successor(action)
                value, _ = self.alphabeta(successor, alpha, beta, depth_limit, current_depth + 1)

                if value < min_value:
                    min_value = value
                    best_action = action

                beta = min(beta, value)

                if beta <= alpha:
                    break

            return min_value, best_action
