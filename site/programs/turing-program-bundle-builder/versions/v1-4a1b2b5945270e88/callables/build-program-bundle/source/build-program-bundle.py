def build_program_bundle(
    source: str,
    destination: str | Path,
    *,
    source_filename: str = "program.py",
    entrypoint: str | None = None,
    title: str | None = None,
    slug: str | None = None,
    probes: Mapping[str, Any] | None = None,
    include_backends: bool = True,
    include_mathematics: bool = True,
) -> ProgramBundle:
    """Compile source and atomically publish its complete versioned bundle."""

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    contract = discover_source_contract(
        source, entrypoint=entrypoint, title=title, slug=slug, probes=probes
    )
    version, source_digest = _content_version(source, contract)
    destination = resolve_publish_root(destination)
    versions = destination / "site" / "programs" / contract.slug / "versions"
    final_directory = versions / version
    if final_directory.is_dir() and (final_directory / "bundle.json").is_file():
        return load_program_bundle(final_directory)
    versions.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=versions))
    compile_log = io.StringIO()
    try:
        from ..common.tensors.accelerator_backends.aot_compile import compile_ast_aot
        from ..common.tensors.topological_reducer import reduce_abstract_tensor_topology
        from ..transmogrifier.graph.graph_express2 import ProcessGraph
        from .backend_sources import collect_backend_sources
        from .fused_program_wasm_backend import emit_wasm_module, required_steps
        from .shell_telemetry import TelemetryChannel, summarize_process_graph
        from .sympy_math_renderer import render_reduced_program_mathematics
        from .wasm_html_shell import emit_html_shell

        parameter_names = _entrypoint_parameters(source, contract.entrypoint)
        feeds = {
            name: _probe_value(contract.feeds.get(name), contract.probe_size)
            for name in parameter_names
        }
        channel = TelemetryChannel(name=f"program:{contract.slug}")
        with contextlib.redirect_stdout(compile_log), contextlib.redirect_stderr(compile_log):
            graph = ProcessGraph(materialize_memory=False)
            graph.build_from_ast(ast.parse(source))
            reduce_abstract_tensor_topology(graph)
            aot = compile_ast_aot(
                source,
                contract.entrypoint,
                feeds,
                backend=contract.backend,
                remove_loops=contract.remove_loops,
                unroll_limit=contract.unroll_limit,
                precompile_only=True,
            )
            program = getattr(aot.compiled_shell_program, "program", aot.compiled_shell_program)
            module = emit_wasm_module(
                program, name=contract.entrypoint, dtype="float64"
            )
        if not module.complete:
            raise RuntimeError(module.shortfall_report())

        wasm_relative = Path("wasm") / f"{module.name}.wasm"
        wasm_path = temporary / wasm_relative
        wasm_path.parent.mkdir(parents=True, exist_ok=True)
        wasm_path.write_bytes(module.binary)

        sources = None
        if include_backends:
            with contextlib.redirect_stdout(compile_log), contextlib.redirect_stderr(compile_log):
                sources = collect_backend_sources(
                    aot,
                    numerical_name=contract.entrypoint,
                    control_name=f"{contract.entrypoint}_control",
                    channel=channel,
                    wasm_source=module.source,
                    program=program,
                )
        published_sources = _write_sources(
            temporary, source, Path(source_filename).name, sources
        )

        mathematics = None
        math_error = ""
        if include_mathematics:
            try:
                document = render_reduced_program_mathematics(
                    program,
                    input_names=parameter_names,
                    program_name=contract.entrypoint,
                )
                payload = document.to_mapping()
                relative = Path("math") / "sympy-process-model.json"
                path = temporary / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(payload, separators=(",", ":")), encoding="utf-8"
                )
                mathematics = {
                    key: value for key, value in payload.items() if key != "equations"
                }
                mathematics.update({
                    "url": relative.as_posix(),
                    "filename": relative.name,
                    "bytes": path.stat().st_size,
                })
            except Exception as error:  # page generation survives optional projection refusal
                math_error = f"{type(error).__name__}: {error}"

        entry = module.api.entry_points[0]
        contiguous = {
            "name": module.name,
            "url": wasm_relative.as_posix(),
            "entry": module.api.entry,
            "inputs": [item.name for item in entry.parameters if item.role == "input"],
            "outputs": [item.name for item in entry.parameters if item.role == "output"],
            "value_type": module.api.metadata.get("value_type", "f64"),
            "element_bytes": module.api.metadata.get("element_bytes", 8),
            "memory_export": module.api.metadata.get("memory_export", "memory"),
            "reserved_bytes": module.api.metadata.get("reserved_bytes", 0),
            "operation_count": len(required_steps(program)),
        }
        route = f"/site/programs/{contract.slug}/versions/{version}/"
        shell = emit_html_shell(
            module.api,
            name="index",
            telemetry=channel,
            process_graph=summarize_process_graph(graph),
            origin_source="",
            feed_expressions=contract.feed_expressions,
            build_parameters={
                "bundle schema": BUNDLE_SCHEMA,
                "content version": version,
                "source SHA-256": source_digest,
                "steps": len(required_steps(program)),
            },
            default_width=contract.width,
            default_height=contract.height,
            backend_sources=published_sources,
            mathematics=mathematics,
            map_ir=aot.map_ir,
            class_graph={
                "modules": [],
                "variants": {},
                "contiguous": contiguous,
                "runtime_version": BUILDER_VERSION,
            },
            resource_route=route,
        )
        page_path = shell.write(temporary)

        log_text = compile_log.getvalue().strip()
        if log_text:
            log_path = temporary / "build" / "compiler.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(log_text + "\n", encoding="utf-8")

        manifest = {
            "schema": BUNDLE_SCHEMA,
            "layout_version": BUNDLE_LAYOUT_VERSION,
            "program": {
                "slug": contract.slug,
                "title": contract.title,
                "entrypoint": contract.entrypoint,
            },
            "version": {
                "id": version,
                "source_sha256": source_digest,
                "builder": BUILDER_VERSION,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "page": {"path": "index.html", "url": route + "index.html"},
            "source": {
                "path": f"source/python_source/{Path(source_filename).name}",
                "filename": Path(source_filename).name,
            },
            "compiler": {
                "backend": contract.backend,
                "remove_loops": contract.remove_loops,
                "unroll_limit": contract.unroll_limit,
                "mathematics_error": math_error,
            },
            "artifacts": _artifact_inventory(temporary),
        }
        manifest_path = temporary / "bundle.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if final_directory.exists():
            existing = load_program_bundle(final_directory)
            shutil.rmtree(temporary)
            return existing
        temporary.replace(final_directory)
        return load_program_bundle(final_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise