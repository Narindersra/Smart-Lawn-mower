from .navigation_types import NavigationState


class NavigationStateMachine:
    """
    Manages valid navigation state transitions.

    The state machine prevents invalid navigation states from being
    entered and provides explicit handling for emergency-stop recovery.
    """

    def __init__(self):
        """Initialize the navigation state machine in the IDLE state."""
        self.state = NavigationState.IDLE

    def transition_to(
        self,
        new_state: NavigationState,
    ) -> None:
        """
        Transition to a new navigation state.

        Args:
            new_state: Navigation state to transition into.

        Raises:
            ValueError: If the requested transition is not allowed.
        """

        # No transition is required when the requested state is already
        # the current state.
        if new_state == self.state:
            return

        if not self._is_valid_transition(new_state):
            raise ValueError(
                f"Invalid navigation transition: "
                f"{self.state.value} -> {new_state.value}"
            )

        self.state = new_state

    def resume_from_emergency_stop(self) -> None:
        """
        Resume navigation after an emergency stop.

        Emergency-stop recovery is intentionally limited to transitioning
        from EMERGENCY_STOP back to NAVIGATING.
        """
        if self.state == NavigationState.EMERGENCY_STOP:
            self.state = NavigationState.NAVIGATING

    def _is_valid_transition(
        self,
        new_state: NavigationState,
    ) -> bool:
        """
        Determine whether a navigation state transition is allowed.

        Args:
            new_state: State being requested.

        Returns:
            True if the transition is valid; otherwise False.
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

        Returns:
            Current NavigationState.
        """
        return self.state