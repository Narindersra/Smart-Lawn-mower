from .navigation_types import NavigationState


class NavigationStateMachine:
    """
    Manages valid navigation state transitions.
    """

    def __init__(self):
        self.state = NavigationState.IDLE

    def transition_to(
        self,
        new_state: NavigationState,
    ) -> None:
        """
        Transition to a new navigation state.
        """

        if new_state == self.state:
            return

        if not self._is_valid_transition(new_state):
            raise ValueError(
                f"Invalid navigation transition: "
                f"{self.state.value} -> {new_state.value}"
            )

        self.state = new_state

    def _is_valid_transition(
        self,
        new_state: NavigationState,
    ) -> bool:
        """
        Determine whether a state transition is allowed.
        """

        valid_transitions = {
            NavigationState.IDLE: {
                NavigationState.NAVIGATING,
                NavigationState.EMERGENCY_STOP,
            },

            NavigationState.NAVIGATING: {
                NavigationState.IDLE,
                NavigationState.AVOIDING,
                NavigationState.REPLANNING,
                NavigationState.GOAL_REACHED,
                NavigationState.EMERGENCY_STOP,
            },

            NavigationState.AVOIDING: {
                NavigationState.IDLE,
                NavigationState.NAVIGATING,
                NavigationState.REPLANNING,
                NavigationState.EMERGENCY_STOP,
            },

            NavigationState.REPLANNING: {
                NavigationState.IDLE,
                NavigationState.NAVIGATING,
                NavigationState.GOAL_REACHED,
                NavigationState.EMERGENCY_STOP,
            },

            NavigationState.GOAL_REACHED: {
                NavigationState.IDLE,
                NavigationState.NAVIGATING,
                NavigationState.EMERGENCY_STOP,
            },

            NavigationState.EMERGENCY_STOP: {
                NavigationState.IDLE,
            },
        }

        return new_state in valid_transitions[self.state]

    def get_state(self) -> NavigationState:
        """
        Return the current navigation state.
        """

        return self.state
    