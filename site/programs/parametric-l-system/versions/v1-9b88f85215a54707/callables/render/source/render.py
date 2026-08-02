def render(
    t: int = 0,
    width: int = 512,
    height: int = 512,
    generations: int = 4,
):
    """Return one RGB frame from a preset selected by the frame counter."""

    from PIL import Image, ImageDraw

    presets = ("hilbert", "peano", "dragon", "koch", "sierpinski", "plant")
    preset_name = presets[int(t) % len(presets)]
    system = ParametricLSystem.preset(preset_name)
    trace = system.trace(int(generations))
    image = Image.new("RGB", (int(width), int(height)), (3, 7, 18))
    draw = ImageDraw.Draw(image)
    if trace.segments:
        min_x, min_y, max_x, max_y = trace.bounds
        span_x = max(max_x - min_x, 1e-12)
        span_y = max(max_y - min_y, 1e-12)
        padding = max(2, min(width, height) // 20)
        scale = min(
            (width - padding * 2) / span_x,
            (height - padding * 2) / span_y,
        )
        offset_x = (width - span_x * scale) * 0.5 - min_x * scale
        offset_y = (height - span_y * scale) * 0.5 + max_y * scale
        last_segment = max(1, len(trace.segments) - 1)
        for index, segment in enumerate(trace.segments):
            phase = index / last_segment
            color = (
                int(127.5 + 127.5 * math.cos(math.tau * phase)),
                int(127.5 + 127.5 * math.cos(math.tau * (phase + 0.21))),
                int(127.5 + 127.5 * math.cos(math.tau * (phase + 0.43))),
            )
            draw.line(
                (
                    segment.start[0] * scale + offset_x,
                    offset_y - segment.start[1] * scale,
                    segment.end[0] * scale + offset_x,
                    offset_y - segment.end[1] * scale,
                ),
                fill=color,
                width=max(1, min(4, int(round(segment.width)))),
            )
    red, green, blue = image.split()
    return red.tobytes(), green.tobytes(), blue.tobytes()