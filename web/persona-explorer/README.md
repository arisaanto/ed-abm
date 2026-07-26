# Part 3 Persona Explorer

Static HTML/CSS/JavaScript interface for browsing Part 3 persona appraisals by
scenario and spatial condition. It is suitable for GitHub Pages and has no
build step or runtime dependency.

Public interface: <https://arisaanto.github.io/ed-abm/>

## Local preview

From this directory:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000>.

## Static paper fallback

The same page can render a reproducible print card for one exact study view.
The URL records the persona, scenario, condition, and qualitative-response
index; adding `paper=1` hides the interactive navigation without creating a
second presentation system. For example:

```text
http://localhost:8000/?persona=team_connector&scenario=normal_load&condition=both&response=1&paper=1
```

Use the browser's Print command to save that selected view as PDF. The normal
interactive interface is unchanged when `paper=1` is absent. Static cards use
the same verified observed appraisal results as the interactive view.

## Results contract

`data/persona-results.json` is the only study-data input. Its current records
are verified synthetic appraisal results generated from the completed Part 3
study. They are not human participant data or psychological measurements. Each
record is keyed by:

```text
persona + scenario + condition
```

Every result preserves the same keys and provides:

- four 1–7 experience scores;
- the aggregate sample size;
- three exact qualitative excerpts from a representative appraisal bundle;
- `meta.status: "observed"`.

OCEAN values are a presentation-only crosswalk from preregistered workplace
priors. They are not measured psychometric scores and do not drive the model.

Each record provides a supportive feature, a difficult feature, and a
counterfactual change from one complete synthetic interview bundle. The bundle
is selected nearest the cell-median fit after deterministic language and
grounding screens. These are synthetic interviews, not Zurich ED staff
testimony.

Regenerate the complete Part 3 findings package and observed explorer JSON
with:

```bash
python3 scripts/build/build_part3_figures_tables.py
```

The build refuses incomplete appraisal or ablation inputs. The four displayed
values map directly to overall person-space fit, team awareness, task
continuity, and spatial legibility. No composite score is constructed.

The interface self-hosts Press Start 2P, Silkscreen, and VT323 under the SIL
Open Font License 1.1; all three font licenses are included in `assets/fonts/`.
