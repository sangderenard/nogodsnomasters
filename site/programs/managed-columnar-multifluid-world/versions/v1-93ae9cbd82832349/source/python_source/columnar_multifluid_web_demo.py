
TURING_PAGE = {
    "entrypoint": "columnar_multifluid_rgb_step",
    "title": "Managed Columnar Multifluid World",
    "slug": "managed-columnar-multifluid-world",
    "width": 384,
    "height": 268,
    "probe_size": 16,
    "feeds": {
        "column_x": 0.5,
        "column_y": 0.5,
        "rest_surface": 1.0,
        "displacement": 0.0,
        "displacement_velocity": 0.0,
        "managed_time": 0.0,
        "dt": 0.025,
        "ink_red": 0.0,
        "ink_yellow": 0.0,
        "ink_green": 0.0,
        "ink_cyan": 0.0,
        "ink_blue": 0.0,
        "ink_magenta": 0.0
    },
    "feed_expressions": {
        "column_x": "(x + 0.5) * 10.0 / w",
        "column_y": "(y + 0.5) * 7.0 / h",
        "rest_surface": "1.15 + 3.1 * Math.exp(-16.0 * (((x + 0.5) / w - 0.62) ** 2 + ((y + 0.5) / h - 0.52) ** 2))",
        "displacement": "0.0",
        "displacement_velocity": "0.0",
        "managed_time": "0.0",
        "dt": "0.025",
        "ink_red": "0.0",
        "ink_yellow": "0.0",
        "ink_green": "0.0",
        "ink_cyan": "0.0",
        "ink_blue": "0.0",
        "ink_magenta": "0.0"
    },
    "state_feedback": {
        "displacement": "next_displacement",
        "displacement_velocity": "next_velocity",
        "managed_time": "next_time",
        "ink_red": "next_ink_red",
        "ink_yellow": "next_ink_yellow",
        "ink_green": "next_ink_green",
        "ink_cyan": "next_ink_cyan",
        "ink_blue": "next_ink_blue",
        "ink_magenta": "next_ink_magenta"
    },
    "render_fps": 30.0,
    "autostart": True,
    "backend": "c",
    "remove_loops": True
}


def columnar_multifluid_rgb_step(
    column_x,
    column_y,
    rest_surface,
    displacement,
    displacement_velocity,
    managed_time,
    dt,
    ink_red,
    ink_yellow,
    ink_green,
    ink_cyan,
    ink_blue,
    ink_magenta,
):
    """One Python-owned managed tick and its three RGB preview planes."""

    next_time = managed_time + dt
    player_x = 5.0 + 3.15 * (next_time * 0.72).sin()
    player_y = 3.5 + 2.05 * (next_time * 1.07 + 1.5707963267948966).sin()
    distance_squared = (
        (column_x - player_x) * (column_x - player_x)
        + (column_y - player_y) * (column_y - player_y)
    )
    entity_interior = (
        (0.52 - (column_x - player_x).abs()).maximum(0.0)
        .minimum((0.52 - (column_y - player_y).abs()).maximum(0.0))
        / 0.52
    )
    load = (-distance_squared / (2.0 * 1.35 * 1.35)).exp()
    target = -0.42 * load - 0.22 * entity_interior * entity_interior
    acceleration = (
        20.0 * (target - displacement) - 8.0 * displacement_velocity
    )
    next_velocity = displacement_velocity + acceleration * dt
    next_displacement = displacement + next_velocity * dt
    surface = rest_surface + next_displacement
    height = ((surface - 0.5) / 5.0).maximum(0.0).minimum(1.0)
    compression = (-next_displacement / 0.42).maximum(0.0).minimum(1.0)
    motion = next_velocity.abs().minimum(1.0)

    base_red = (
        186.0 + 27.0 * height - 34.0 * compression + 54.0 * load
    ).maximum(0.0).minimum(255.0)
    base_green = (
        220.0 + 18.0 * height - 21.0 * compression + 30.0 * load
        + 8.0 * motion
    ).maximum(0.0).minimum(255.0)
    base_blue = (
        232.0 + 16.0 * height + 15.0 * compression + 20.0 * load
    ).maximum(0.0).minimum(255.0)

    hue = next_time * 0.42
    source_red = (-distance_squared / (2.0 * 0.44 * 0.44)).exp()
    source_yellow = (-distance_squared / (2.0 * 0.48 * 0.48)).exp()
    source_green = (-distance_squared / (2.0 * 0.52 * 0.52)).exp()
    source_cyan = (-distance_squared / (2.0 * 0.56 * 0.56)).exp()
    source_blue = (-distance_squared / (2.0 * 0.60 * 0.60)).exp()
    source_magenta = (-distance_squared / (2.0 * 0.64 * 0.64)).exp()
    weight_red = hue.cos().maximum(0.0)
    weight_yellow = (hue - 1.0471975511965976).cos().maximum(0.0)
    weight_green = (hue - 2.0943951023931953).cos().maximum(0.0)
    weight_cyan = (hue - 3.141592653589793).cos().maximum(0.0)
    weight_blue = (hue - 4.1887902047863905).cos().maximum(0.0)
    weight_magenta = (hue - 5.235987755982989).cos().maximum(0.0)
    next_ink_red = (
        ink_red * (-0.050 * dt).exp() + 2.8 * dt * source_red * weight_red
    ).minimum(1.0)
    next_ink_yellow = (
        ink_yellow * (-0.054 * dt).exp()
        + 2.8 * dt * source_yellow * weight_yellow
    ).minimum(1.0)
    next_ink_green = (
        ink_green * (-0.058 * dt).exp() + 2.8 * dt * source_green * weight_green
    ).minimum(1.0)
    next_ink_cyan = (
        ink_cyan * (-0.062 * dt).exp() + 2.8 * dt * source_cyan * weight_cyan
    ).minimum(1.0)
    next_ink_blue = (
        ink_blue * (-0.066 * dt).exp() + 2.8 * dt * source_blue * weight_blue
    ).minimum(1.0)
    next_ink_magenta = (
        ink_magenta * (-0.070 * dt).exp()
        + 2.8 * dt * source_magenta * weight_magenta
    ).minimum(1.0)
    ink_total = (
        next_ink_red + next_ink_yellow + next_ink_green
        + next_ink_cyan + next_ink_blue + next_ink_magenta
    ).maximum(1.0e-6)
    ink_alpha = ink_total.minimum(0.88)
    ink_color_red = 255.0 * (
        next_ink_red + next_ink_yellow + next_ink_magenta
    ) / ink_total
    ink_color_green = 255.0 * (
        next_ink_yellow + next_ink_green + next_ink_cyan
    ) / ink_total
    ink_color_blue = 255.0 * (
        next_ink_cyan + next_ink_blue + next_ink_magenta
    ) / ink_total
    red = base_red * (1.0 - ink_alpha) + ink_color_red * ink_alpha
    green = base_green * (1.0 - ink_alpha) + ink_color_green * ink_alpha
    blue = base_blue * (1.0 - ink_alpha) + ink_color_blue * ink_alpha
    entity_glow = entity_interior * entity_interior
    red = red * (1.0 - entity_glow) + 245.0 * entity_glow
    green = green * (1.0 - entity_glow) + 252.0 * entity_glow
    blue = blue * (1.0 - entity_glow) + 255.0 * entity_glow
    return (
        red,
        green,
        blue,
        next_displacement,
        next_velocity,
        next_time,
        next_ink_red,
        next_ink_yellow,
        next_ink_green,
        next_ink_cyan,
        next_ink_blue,
        next_ink_magenta,
    )
