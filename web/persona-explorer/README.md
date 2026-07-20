# Part 3 Persona Explorer

Static HTML/CSS/JavaScript interface for browsing Part 3 persona appraisals by
scenario and spatial condition. It is suitable for GitHub Pages and has no
build step or runtime dependency.

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
interactive interface is unchanged when `paper=1` is absent. Static cards
should only be exported after `persona-results.json` is replaced with verified
observed appraisal results.

## Results contract

`data/persona-results.json` is the only study-data input. Its current records
are explicitly marked `preview_placeholder` and are not scientific results.
Each record is keyed by:

```text
persona + scenario + condition
```

The production export should preserve the same keys and provide:

- four 1–7 experience scores;
- the aggregate sample size;
- up to three selected or synthesized qualitative responses grounded in the
  appraisal outputs, provided as a `quotes` array;
- `meta.status: "observed"` after verification.

OCEAN values are a presentation-only crosswalk from preregistered workplace
priors. They are not measured psychometric scores and do not drive the model.

Preview and observed records provide up to three concise `quotes`. Answers
begin with the experienced consequence rather than repeating the question or
announcing scenario and condition labels.

After appraisal verification and manual review, generate the observed JSON
with:

```bash
python3 scripts/export_part3_persona_explorer_data.py \
  --analysis-dir /path/to/verified_appraisal_analysis \
  --output web/persona-explorer/data/persona-results-observed.json
```

The exporter refuses incomplete review coverage. Exactly three responses per
persona/scenario/condition cell must be explicitly selected in
`interview_manual_review_sample.csv`; selected responses must pass grounding,
direct-answer, unobtrusive-context, claim-layer, tradeoff, and naturalness
review. The four displayed values map directly to overall person-space fit,
team awareness, task continuity, and spatial legibility. No composite score is
constructed.

The interface self-hosts Press Start 2P, Silkscreen, and VT323 under the SIL
Open Font License 1.1; all three font licenses are included in `assets/fonts/`.
