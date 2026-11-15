from src.grid_environment import GridWorld


def test_start_state():
    env = GridWorld()
    assert env.reset() == (2, 1)


def test_wall_collision_keeps_state():
    env = GridWorld()
    env.state = (1, 1)
    transition = env.step("N")
    assert transition.next_state == (1, 1)
    assert transition.reward == -1
    assert transition.done is False


def test_jump_transition():
    env = GridWorld()
    env.state = (2, 3)
    transition = env.step("E")
    assert transition.next_state == (4, 4)
    assert transition.reward == 5


def test_goal_transition():
    env = GridWorld()
    env.state = (5, 4)
    transition = env.step("E")
    assert transition.next_state == (5, 5)
    assert transition.reward == 10
    assert transition.done is True


def test_goal_is_absorbing_and_does_not_repeat_terminal_reward():
    env = GridWorld()
    env.state = (5, 4)
    first = env.step("E")
    assert first.next_state == (5, 5)
    assert first.reward == 10
    assert first.done is True

    for action in GridWorld.ACTIONS:
        later = env.step(action)
        assert later.next_state == (5, 5)
        assert later.reward == 0
        assert later.done is True
        assert env.state == (5, 5)


def test_reset_reopens_episode_after_terminal_goal():
    env = GridWorld()
    env.state = (5, 4)
    assert env.step("E").done is True
    assert env.reset() == (2, 1)
    next_move = env.step("E")
    assert next_move.next_state == (2, 2)
    assert next_move.reward == -1
    assert next_move.done is False
