# Meowdoku / Nodus work tracker — 2026-11-07

This is the current queue for the requests made in this chat. "Live" means checked on HTTPS port 8766. Worktrees and branches named here are separate from canonical `C:/dev/Powershell/turing` main. Keep unrelated dirty compiler and chess files in main intact.

| Request | Owner | State / next gate |
|---|---|---|
| Audit Python server, compile choices, client paths, shader/Wasm/GPU routes, bespoke human/AI steps, and Nodus migration from systems/graph/chaos perspective | Root | Audits saved at this level, in `speaktome`, `nodus`, `turing`, and Meowdoku. Revisit against new case-motion and game code before planning Nodus extraction. |
| Trace recent Claude branches and continue their work | Root | Layout stages and handoff are integrated in `turing-claude-integration-20261107`; not landed because main's dirty `search.py` intersects its move and product digests change. Continue from `examples/meowdoku/CLAUDE_BRANCH_CONTINUATION_2026-11-07.md`. |
| Client and server logs on a page | Root | Live on `/log.html`; server restarted on 8766 and `/api/log` checked. |
| Browser settings reset button | Root | Restored and live; do not remove. It is distinct from restarting the Python server. |
| Restart live server | Root | Completed by stopping and relaunching recorded Python configuration on 8766. New process and `/api/ready` checked. |
| Studio can select every fountain size | Studio agent, root integration | Live: small, 4k, 16k, 64k, 256k, 1M, 4M, 10M; availability checked against deployed manifest. |
| Large simulations, including about 200k particles, with visual-only recording when full trace is too expensive | Studio agent, root integration | 256k product is deployed; bounded visual-only capture is live. Real browser/device performance at 256k remains to be measured. |
| Rich Phong material catalogue, several distinct metals | Material agent, root integration | 19 catalogue additions, 11 metals, 22 exported material IDs live. |
| Cat-eye marble layered mesh and designer | Marble agent, root integration | Designer and layered mesh live. Verify visual result in browser. |
| Playable Marble Madness course | Root integration, Marble agent | Commit `11def8a6` in `turing-marble-play-integration-20261107` reconciles the game with the shared 3D toy; focused Python tests pass. Scratch HTTPS 8767 lists it in the game catalog and `/api/new` returns the 4k toy manifest. Browser play and live landing remain. |
| Finish 2D/3D hourglass toy | Root | 3D `sdf_toy` code in `turing-claude-integration-20261107`; scratch WebGPU small and 4k built. Branch merged latest main and passed 27 focused Python tests. Scratch 8767 server is building background WebGPU families with `--no-wasm`; live product deployment and 2D peg-filter clog remain. |
| WebGPU build now; Wasm only on actual fallback need or server idle | Root | WebGPU toy small and 4k built in scratch. No Wasm build ran. On-demand/idle Wasm scheduling remains to implement; do not eagerly compile it. |
| Flip beside Rock/Rotate; choose axis; move actual case; share mechanism with Studio hourglass loading | Studio agent, root review | Commit `33af3c2b` in isolated Studio branch implements a finite case turn shared by physics, case rendering, and loading recording, respecting the existing law cap. Existing laws support only the case-normal axis (world Y); X/Z require law work. Browser GPU validation and landing remain. |
| Loading screen visible before Meowdoku, with brief indication for each real startup step | Material agent, root integration | Done on live 8766: browser sampling showed splash through the 28 MB download and first rendered board frame, then clean hide at about 8.4 seconds. Tiny `#status` text under nav is a separate layout artifact. |
| Dedicated close-up marble designer with full Phong materials, color choices, and rerollable dye shape | Marble agent, root integration | Live at `/marble_studio.html` on 8766; headless Chrome screenshot confirmed the layered marble and controls render. Independent of playable course and toy product; no Wasm build. |
| Reshape cat-eye dye into separate lit ribbons; more triangles, shaping controls, extruded tips, twist along curve, later compact game mesh | Marble agent, root integration | Separate closed ribbons with taper/twist and 97,280-triangle studio preview are live. Default framing is being improved; game mesh reduction remains. |
| Marble finishes and design families: phase placement, confetti roll, partial enamel, opaque/clear shell, checkerboard lighting scene, no decoration, galaxy, uniform confetti, tubular spirals and swooping rings | Marble agent | In progress in isolated branch. Use real geometry and exported Pluck material/shader tables; do not claim ray tracing for the current Phong transmission path. |
| Porcelain and outer-painted watercolor cat-eye marbles with color bands | Marble agent | In progress in isolated branch. Model the exterior bands as actual painted material regions with controllable color, width and placement. |
| Marble physical properties from final geometry | Marble agent | In progress: compute component volumes, displaced glass volume, total mass, center of mass and rotational inertia from closed meshes with documented densities. Tie the future game representation to the baked design without changing its physics by guesswork. |
| Remove the long instruction block from the visible game layout while keeping it in page source | Root | Live: `#help` has `hidden` and `aria-hidden`, with no visible layout styles. |
| Shift server backbone to Nodus KPN, with clearly marked CPython-only nodes | Root / future Nodus lane | Architecture audit saved. Implementation has not begun; first step is to map server duties to graph nodes and identify existing Nodus Python hosting and runtime boundaries. |

## Order and integration gates

1. Recheck the live loading panel through the entire startup after the first-frame hold fix. Review the marble studio's material appearance and controls with the user.
2. Review and land the shared case Flip implementation only after its physical timing, 2D/3D law inputs, Studio preview, and case mesh agree. A law/source change may require approved WebGPU rebuilds; Wasm stays lazy.
3. Reconcile the Claude layout branch with main's dirty chess file without discarding local work. Rebuild affected products in scratch, then deploy the 3D toy and browser-test the marble course. The marble game commit depends on this gate.
4. Measure 256k recording on the target browser/device; adjust the existing budget/path only from observed failures.
5. Convert the Nodus audit into a graph of service responsibilities and begin Nodus ownership of one real server duty at a time, retaining explicit CPython nodes where lowering is unavailable.

## Evidence locations

- Root and per-repository `NODUS_MEOWDOKU_SERVICE_BACKBONE_AUDIT_2026-11-07.md` files.
- `speaktome/AGENTS/experience_reports/` for cross-repository receipts.
- `turing-claude-integration-20261107/examples/meowdoku/CLAUDE_BRANCH_CONTINUATION_2026-11-07.md` for layout and toy gates.
- `turing-marble-game-20261107/examples/meowdoku/MARBLE_MADNESS_HANDOFF.md` for the playable course dependency and limitations.
