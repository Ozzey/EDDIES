# Project website

The project page is `docs/index.html` with its stylesheet `docs/styles.css`, a small script `docs/script.js`
(ES module, no framework, no build step) and `docs/favicon.svg`. It links to `docs/paper.pdf`,
`docs/supplementary.pdf` and to media under `../assets/` with relative paths, so the same files work from a
local checkout and on GitHub Pages.

## What the page expects to find

| Reference in `index.html`                         | Produced by                         |
|---------------------------------------------------|-------------------------------------|
| `paper.pdf`, `supplementary.pdf`                  | the paper build (copied into docs/) |
| `../assets/architectures/eddies_architecture.svg` | the animated architecture SVG (14 s loop, five stages, CSS keyframes inside the SVG). The page embeds it with `<object>`; `script.js` adds a **Pause animation / Play animation** button that calls the SVG's `pauseAnimations()` / `unpauseAnimations()` and respects `prefers-reduced-motion` |
| `../assets/videos/sim/<task>.mp4` (ten registry task names) | overlaid MuJoCo renders |
| `../assets/videos/real/{stacking,perturbation,rgb_feed,two_targets}.mp4` | real-Franka clips |
| `../assets/videos/supplementary.mp4`              | supplementary video |
| `../assets/videos/previews/<task>.jpg`, `previews/isaac/<task>.jpg`, `previews/{stacking,perturbation,rgb_feed,supplementary}.jpg` | poster frames: stills of seed 2000 for the ten tasks, frames of the clips for the real-robot and supplementary posters |
| `../assets/results/paper/res_{cert,gate,heldout}.svg` | Figs. 4, 5, 7 of the paper converted with `pdftocairo -svg` |
| `../assets/results/real_setup.png`, `real_stop/*.jpg` | Fig. 8 of the paper and its crops |

Video tags carry a poster image, so the page shows the still until a clip loads; a clip that cannot be loaded
keeps its still and a "Video unavailable" label instead of a broken player.

## What the script does

`script.js` only adds behaviour that degrades gracefully without it: the "More research" menu, the demo tabs
(real Franka / G1 / LEAP hand) with one play button, the play overlays on the demo cards (one clip plays at a time,
clips pause when the tab is hidden), the architecture pause button, and the results tabs. Nothing on the page is
rendered by the script; every section, table and figure is static HTML.

## Local preview

```bash
cd EDDIES
python -m http.server 8000
# open http://localhost:8000/docs/
```

Or open `docs/index.html` directly in a browser; everything is relative (the `<object>` embed of the SVG and the
module script need the http server in some browsers when opened from `file://`).

## Editing

* Numbers in the tables are copied from the paper (`docs/paper.pdf`, Tables 1-3); keep them in sync with the paper.
* Colours are CSS custom properties at the top of `styles.css` (`--ink`, `--accent`, ... mapped to the paper's
  scheme: navy `#221b45`, blue `#2e90ef`, orange `#f39a2b`, violet `#6a1fd6`, red `#d6452b`), the same scheme as
  the SVGs and the video overlays.
* The architecture animation lives entirely in the SVG; the page does not synchronise any caption with it, so the
  loop length can change without touching `index.html`.

## Hosting on the personal site

The page is published at https://ozzey.github.io/papers/eddies/ from the `ozzey.github.io` repository. Export the folder with

```sh
python scripts/export_site.py /path/to/ozzey.github.io/papers/eddies
```

which copies the page, its stylesheet/script, the PDFs and the small assets, and rewrites the video sources to stream from this repository on GitHub (`raw.githubusercontent.com/Ozzey/EDDIES/main/assets/videos/...`), so the site repository stays small. Pass `--copy-videos` to copy the mp4s instead.
