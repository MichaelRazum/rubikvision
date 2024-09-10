from rubikvision.cube_solver import CubeState, ActionExecutor


def test_get_next_action():
    cube_state = CubeState()
    upper= ['U' for _ in range(9)]
    left  = ['L' for _ in range(9)]
    front = ['F' for _ in range(9)]
    cube_state.update(upper=upper, left=left, front=front)

    executor = ActionExecutor()
    action = executor.get_next_action(cube_state=cube_state, target=cube_state)
    assert action is None
    target_state = cube_state.copy_from_state(current_upper='!' + cube_state.current_upper,
                                              current_front='!' + cube_state.current_front,
                                              current_left='!' + cube_state.current_left)

    action = executor.get_next_action(cube_state=cube_state, target=target_state)
    assert action == 'flip'

    cube_state = cube_state.copy_from_state(current_upper='L',
                                            current_front='F',
                                            current_left='W')

    action = executor.get_next_action(cube_state=cube_state, target=target_state)
    assert action == 'flip'

    cube_state = cube_state.copy_from_state(current_upper='W',
                                            current_front='F',
                                            current_left='Z')

    action = executor.get_next_action(cube_state=cube_state, target=target_state)
    assert action == 'rotate_left'

def test_executor_solve_step():
    executor = ActionExecutor()
    state_1 = CubeState(current_upper='white', current_left='green', current_front='red',
                        upper=['white', 'yellow', 'white', 'orange', 'yellow', 'blue', 'white', 'yellow', 'yellow'],
                        left=['orange', 'green', 'orange', 'red', 'orange', 'blue', 'blue', 'red', 'red'],
                        front=['green', 'red', 'green', 'white', 'blue', 'green', 'yellow', 'blue', 'orange'],
                        down=['green', 'orange', 'blue', 'blue', 'white', 'red', 'yellow', 'white', 'red'],
                        right=['orange', 'yellow', 'red', 'yellow', 'red', 'white', 'yellow', 'green', 'green'],
                        back=['blue', 'orange', 'blue', 'orange', 'green', 'white', 'white', 'green', 'red'])
    solution = "D"
    state_2 = state_1.get_next_cube(solution)
    action = executor.get_next_action(cube_state=state_1, target=state_2, solution=solution)
    assert action == 'rotate_upper_right'

    solution = "D'"
    state_2 = state_1.get_next_cube(solution)
    action = executor.get_next_action(cube_state=state_1, target=state_2, solution=solution)
    assert action == 'rotate_upper_left'

    solution = "U"
    state_2 = state_1.get_next_cube(solution)
    action = executor.get_next_action(cube_state=state_1, target=state_2, solution=solution)
    assert action == 'flip'

    solution = "B"
    state_2 = state_1.get_next_cube(solution)
    action = executor.get_next_action(cube_state=state_1, target=state_2, solution=solution)
    assert action == 'flip'

    solution = "R"
    state_2 = state_1.get_next_cube(solution)
    action = executor.get_next_action(cube_state=state_1, target=state_2, solution=solution)
    assert action == 'rotate_right'

    solution = "L"
    state_2 = state_1.get_next_cube(solution)
    action = executor.get_next_action(cube_state=state_1, target=state_2, solution=solution)
    assert action == 'rotate_left'


