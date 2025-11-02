from agent_base import Agent
from game import GameState
from evaluation import betterEvaluationFunction

class ExpectimaxAgent(Agent):
    def get_action(self, state: GameState, depth=None):
        value, action = self.expectimax(state, depth_limit=depth, current_depth=0)
        if action is None:
            legal = state.get_legal_actions()
            if legal:
                action = legal[0]
        return action

    def expectimax(self, state: GameState, depth_limit, current_depth):
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
                value, _ = self.expectimax(successor, depth_limit, current_depth + 1)

                if value > max_value:
                    max_value = value
                    best_action = action

            return max_value, best_action

        else:
            total_value = 0
            num_actions = len(legal_actions)

            for action in legal_actions:
                successor = state.generate_successor(action)
                value, _ = self.expectimax(successor, depth_limit, current_depth + 1)
                total_value += value

            expected_value = total_value / num_actions if num_actions > 0 else 0
            return expected_value, None