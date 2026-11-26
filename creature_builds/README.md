# creature_builds: every build of the mechanical creature game

An archive of operator-made game builds and the switch for which one the speaktome.dev landing
shows. These are not dev pages: a dev page is user material made by dev-level members and placed
in the site map. Each game page has a folder (`mechanical_creature/`). Each build has its own
folder inside it, `vNN-<label>-<sha8 of the entry page>`, with one commit and one tag
`creature_builds/<page>/<version>` in nogodsnomasters. It holds every file the entry pages load, so the folder can be served as it is.

- Build files are never edited after their commit. A newer build is a new folder.
  Annotations (`LIMITATIONS.md`, `known_limitations`) are added as new commits.
- `VERSION.json`: build date, source folder, notes, known limitations, the sha256 of every
  file, `not_in_archive` (referenced but absent at the source) and `changed_after_build`
  (dependencies whose source copy is newer than the build: copied as they are now).
- `versions.json`: every page, its versions and its `default`. The site's landing page
  shows the default version; `index.html` here lists every version for comparison.
- These are generated pages. Fix the generator
  (`turing/tools/build_mechanical_creature_page.py` and the Studio patch tools), never these copies.

Served on Oracle at `https://speaktome.dev/creature_builds/`; the apex `/` follows `versions.json` `default`.
