# v05-layer-dev-4dc7b488: known limitations

- Hazard 20261110T212000Z: a positive pedal can drive the car backward; tire_reaction_torque inputs are near zero.
- Static hosting (speaktome.dev/pages, 2026-11-26): the game page loads with no missing files, but marble_studio_layer.html needs the Meowdoku server API (/api/whoami, /api/ready, /api/marble_photo/*, /api/marble_objects) and stops at marble_photo.js:710 (null addEventListener). Its marble_*.js dependencies were copied as they are now (see changed_after_build), not as they were at build time.
