def preset(cls, name: str, **overrides: Any) -> "ParametricLSystem":
        """Construct a classic system by name.

        Available names are ``hilbert``, ``peano``, ``dragon``, ``koch``,
        ``sierpinski``, and ``plant``. Any constructor argument can be
        replaced through ``overrides``.
        """

        presets: dict[str, dict[str, Any]] = {
            "hilbert": {
                "axiom": "A",
                "rules": {"A": "+BF-AFA-FB+", "B": "-AF+BFB+FA-"},
                "angle_degrees": 90.0,
            },
            "peano": {
                "axiom": "X",
                "rules": {
                    "X": "XFYFX+F+YFXFY-F-XFYFX",
                    "Y": "YFXFY-F-XFYFX+F+YFXFY",
                },
                "angle_degrees": 90.0,
            },
            "dragon": {
                "axiom": "FX",
                "rules": {"X": "X+YF+", "Y": "-FX-Y"},
                "angle_degrees": 90.0,
            },
            "koch": {
                "axiom": "F--F--F",
                "rules": {"F": "F+F--F+F"},
                "angle_degrees": 60.0,
            },
            "sierpinski": {
                "axiom": "F-G-G",
                "rules": {"F": "F-G+F+G-F", "G": "GG"},
                "angle_degrees": 120.0,
            },
            "plant": {
                "axiom": "X",
                "rules": {"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"},
                "angle_degrees": 25.0,
            },
        }
        key = name.casefold().replace("-", "_").replace(" ", "_")
        aliases = {
            "hilbert_curve": "hilbert",
            "peano_curve": "peano",
            "dragon_curve": "dragon",
            "koch_snowflake": "koch",
            "sierpinski_triangle": "sierpinski",
            "fractal_plant": "plant",
        }
        key = aliases.get(key, key)
        if key not in presets:
            raise ValueError(
                f"unknown preset {name!r}; choose from {', '.join(sorted(presets))}"
            )
        config = {**presets[key], **overrides}
        return cls(**config)