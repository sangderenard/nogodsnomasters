# Springtail playable tires

Recovered matched whole-game artifact from root commit aa0b8de8adc89c6d739f14b874195bf1dd19fd77, with its existing load-dependent tire friction (sensitivity 0.075). No pneumatic simulator or compiled kernels changed. Historical chassis-leveling const reassignment was corrected to let; otherwise the original page is retained, including its older shell and terrain.

SHA256: aa6cb6fb897d24e6bf769ca2dbfde3a0d3d0dbcdd64ddb87fd30173bc0f0d5e2.

Verification: both JS packets parse; all 7 Wasm modules validate and match their hashes. Actual contact Wasm proves load-sensitive traction and zero airborne forces. Full original worker test produces529 finite snapshots, moves over2m under throttle and slows under braking. Actual Chrome QA renders full car/terrain, W accelerates0.5→2.3m/s, V dismounts, then remount and W reaches3.5m/s. Browser braking and WebGPU-specific correctness were not independently asserted.

Published separately through existing server at /mechanical_creature_play.html; localhost trusted HTTPS returns200 with identical bytes. No server restart or newer-page overwrite. External-IP verification blocked by network policy; no public reachability claim. Evidence/repeatable restore and tests: turing/build/dot-playable-tires-20261110. Pneumatic/native-validator and long-duration stability are outside this restoration.
