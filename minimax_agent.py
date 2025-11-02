from agent_base import Agent
from game import GameState

class MinimaxAgent(Agent):
    def get_action(self, state: GameState, depth=None):
        _, action = self.minimax(state)
        return action

    def minimax(self, state, depth_limit=None):
        if state.is_terminal():
            return state.utility(), None

        legal_actions = state.get_legal_actions()

        if state.to_move == 'X':
            max_value = float('-inf')
            best_action = None

            for action in legal_actions:
                successor = state.generate_successor(action)
                value, _ = self.minimax(successor, depth_limit)

                if value > max_value:
                    max_value = value
                    best_action = action

            return max_value, best_action

        else:
            min_value = float('inf')
            best_action = None

            for action in legal_actions:
                successor = state.generate_successor(action)
                value, _ = self.minimax(successor, depth_limit)

                if value < min_value:
                    min_value = value
                    best_action = action

            return min_value, best_action
