
TURING_PAGE = {
    "entrypoint": "dom_surface_step",
    "title": "Elastic DOM Surface",
    "slug": "elastic-dom-surface",
    "width": 960,
    "height": 640,
    "probe_size": 4,
    "feeds": {
        "position_x": {"values": [90.0, 260.0, 430.0, 600.0]},
        "position_y": {"values": [90.0, 150.0, 230.0, 320.0]},
        "velocity_x": 0.0,
        "velocity_y": 0.0,
        "anchor_x": {"values": [90.0, 260.0, 430.0, 600.0]},
        "anchor_y": {"values": [90.0, 150.0, 230.0, 320.0]},
        "extent_x": {"values": [140.0, 150.0, 160.0, 180.0]},
        "extent_y": {"values": [48.0, 64.0, 72.0, 84.0]},
        "pointer_x": 0.0,
        "pointer_y": 0.0,
        "pointer_buttons": 0.0,
        "dt": 0.016666666666666666
    },
    "backend": "c",
    "remove_loops": True
}


def dom_surface_step(
    position_x,
    position_y,
    velocity_x,
    velocity_y,
    anchor_x,
    anchor_y,
    extent_x,
    extent_y,
    pointer_x,
    pointer_y,
    pointer_buttons,
    dt,
):
    half_x = extent_x * 0.5
    half_y = extent_y * 0.5
    inside_x = (pointer_x >= position_x - half_x) * (pointer_x <= position_x + half_x)
    inside_y = (pointer_y >= position_y - half_y) * (pointer_y <= position_y + half_y)
    hovered = inside_x * inside_y
    grabbed = hovered * (pointer_buttons > 0.0)

    target_x = anchor_x + grabbed * (pointer_x - anchor_x)
    target_y = anchor_y + grabbed * (pointer_y - anchor_y)
    acceleration_x = (target_x - position_x) * 42.0 - velocity_x * 8.5
    acceleration_y = (target_y - position_y) * 42.0 - velocity_y * 8.5
    next_velocity_x = (velocity_x + acceleration_x * dt) * 0.997
    next_velocity_y = (velocity_y + acceleration_y * dt) * 0.997
    next_position_x = position_x + next_velocity_x * dt
    next_position_y = position_y + next_velocity_y * dt

    speed_energy = (
        next_velocity_x * next_velocity_x + next_velocity_y * next_velocity_y
    ) * 0.0004
    activity = (hovered * 0.35 + grabbed * 0.65 + speed_energy).minimum(1.0)
    return (
        next_position_x,
        next_position_y,
        next_velocity_x,
        next_velocity_y,
        activity,
    )
