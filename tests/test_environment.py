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
