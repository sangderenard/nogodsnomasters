def interpret(self, word: str) -> TurtleTrace:
        """Interpret an already-derived word as two-dimensional geometry."""

        if not isinstance(word, str):
            raise TypeError("word must be a string")
        x, y = self.initial_position
        heading = math.radians(self.initial_heading_degrees)
        turn_angle = math.radians(self.angle_degrees)
        step = self.step_length
        width = self.width
        stack: list[tuple[float, float, float, float, float, float]] = []
        segments: list[TurtleSegment] = []
        maximum_depth = 0
        min_x = max_x = x
        min_y = max_y = y
        generator = random.Random(self._interpretation_seed())

        def snapshot() -> TurtleState:
            return TurtleState(
                position=(x, y),
                heading_degrees=math.degrees(heading),
                step_length=step,
                turn_angle_degrees=math.degrees(turn_angle),
                width=width,
                branch_depth=len(stack),
            )

        for index, symbol in enumerate(word):
            action = self.actions.get(symbol)
            if action is None:
                continue
            context = TurtleContext(
                symbol=symbol,
                index=index,
                word=word,
                state=snapshot(),
                parameters=self.parameters,
                rng=generator,
            )
            for name in self._action_names(action, context):
                if name in {"draw", "move"}:
                    factor = 1.0 + generator.uniform(
                        -self.length_jitter, self.length_jitter
                    )
                    distance = max(0.0, step * factor)
                    start = (x, y)
                    x += math.cos(heading) * distance
                    y += math.sin(heading) * distance
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)
                    if name == "draw":
                        if len(segments) >= self.max_segments:
                            raise OverflowError(
                                f"interpretation exceeded max_segments={self.max_segments}"
                            )
                        segments.append(
                            TurtleSegment(
                                start=start,
                                end=(x, y),
                                width=width,
                                symbol=symbol,
                                symbol_index=index,
                                branch_depth=len(stack),
                            )
                        )
                elif name in {"left", "right"}:
                    jitter = math.radians(
                        generator.uniform(
                            -self.angle_jitter_degrees,
                            self.angle_jitter_degrees,
                        )
                    )
                    heading += (turn_angle if name == "left" else -turn_angle) + jitter
                elif name == "reverse":
                    heading += math.pi
                elif name == "push":
                    stack.append((x, y, heading, step, turn_angle, width))
                    maximum_depth = max(maximum_depth, len(stack))
                elif name == "pop":
                    if not stack:
                        if self.strict_branches:
                            raise ValueError(f"unmatched branch pop at symbol {index}")
                        continue
                    x, y, heading, step, turn_angle, width = stack.pop()
                elif name == "step_up":
                    step *= self.step_scale
                elif name == "step_down":
                    step /= self.step_scale
                elif name == "angle_up":
                    turn_angle *= self.angle_scale
                elif name == "angle_down":
                    turn_angle /= self.angle_scale
                elif name == "width_up":
                    width *= self.width_scale
                elif name == "width_down":
                    width /= self.width_scale
                elif name != "noop":  # guarded by _action_names
                    raise AssertionError(f"unhandled turtle action {name!r}")

        if stack and self.strict_branches:
            raise ValueError(f"word ended with {len(stack)} unclosed branch(es)")
        return TurtleTrace(
            word=word,
            segments=tuple(segments),
            bounds=(min_x, min_y, max_x, max_y),
            final_state=snapshot(),
            maximum_branch_depth=maximum_depth,
        )