
def symbolic_fluid_advance(state, dt):
    previous_mass = 0.0
    next_mass = 0.0
    max_wave_speed = 0.0
    max_height_violation = 0.0
    max_tracer_violation = 0.0
    height_count = state.height_count
    width_count = state.width_count
    for row in range(height_count):
        north = (row - 1) % height_count
        south = (row + 1) % height_count
        for column in range(width_count):
            west = (column - 1) % width_count
            east = (column + 1) % width_count
            previous_mass = previous_mass + state.height[row, column]
            (
                height_next,
                momentum_x_next,
                momentum_y_next,
                tracer_next,
                velocity_x,
                velocity_y,
                vorticity,
                speed,
                wave_speed,
                height_violation,
                tracer_violation,
            ) = symbolic_fluid_step(
                state.coriolis,
                dt,
                state.dx,
                state.gravity,
                state.height[row, column],
                state.height[row, east],
                state.height[north, column],
                state.height[south, column],
                state.height[row, west],
                state.linear_drag,
                state.minimum_height,
                state.momentum_x[row, column],
                state.momentum_x[row, east],
                state.momentum_x[north, column],
                state.momentum_x[south, column],
                state.momentum_x[row, west],
                state.momentum_y[row, column],
                state.momentum_y[row, east],
                state.momentum_y[north, column],
                state.momentum_y[south, column],
                state.momentum_y[row, west],
                state.tracer[row, column],
                state.tracer_diffusivity,
                state.tracer[row, east],
                state.tracer[north, column],
                state.tracer[south, column],
                state.tracer[row, west],
                state.viscosity,
            )
            state.next_height[row, column] = height_next
            state.next_momentum_x[row, column] = momentum_x_next
            state.next_momentum_y[row, column] = momentum_y_next
            state.next_tracer[row, column] = tracer_next
            next_mass = next_mass + height_next
            max_wave_speed = max(max_wave_speed, wave_speed)
            max_height_violation = max(max_height_violation, height_violation)
            max_tracer_violation = max(max_tracer_violation, tracer_violation)
    state.height = state.next_height + 0.0
    state.momentum_x = state.next_momentum_x + 0.0
    state.momentum_y = state.next_momentum_y + 0.0
    state.tracer = state.next_tracer + 0.0
    mass_error = abs(next_mass - previous_mass) / max(abs(previous_mass), 1.0e-30)
    dt_stable_limit = 0.45 * state.dx / max(max_wave_speed, 1.0e-30)
    metrics = Metrics(
        max_vel=max_wave_speed,
        max_flux=max_wave_speed,
        div_inf=0.0,
        mass_err=mass_error,
        dt_limit=dt_stable_limit,
        error_channels={
            "height_positivity": max_height_violation,
            "tracer_bounds": max_tracer_violation,
        },
    )
    state.last_wave_speed = max_wave_speed + 0.0
    state.last_height_violation = max_height_violation + 0.0
    state.last_tracer_violation = max_tracer_violation + 0.0
    return max_height_violation == 0.0 and max_tracer_violation == 0.0, metrics


def symbolic_fluid_frame(state, targets, controller, frame_duration, dt_initial):
    advanced, dt_next, metrics = run_superstep(
        state,
        frame_duration,
        dt_initial,
        state.dx,
        targets,
        controller,
        symbolic_fluid_advance,
        max_iters=256,
    )
    return (
        state.height,
        state.momentum_x,
        state.momentum_y,
        state.tracer,
        dt_next,
        state.last_wave_speed,
        state.last_height_violation,
        state.last_tracer_violation,
    )
