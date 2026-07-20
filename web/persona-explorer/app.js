const DATA_URL = "data/persona-results.json";

const SCORE_LABELS = {
  overall_experience: "Overall experience",
  coordination_support: "Coordination support",
  focus_support: "Focus support",
  spatial_legibility: "Spatial legibility",
};

const OCEAN_LABELS = {
  openness: "Openness",
  conscientiousness: "Conscientiousness",
  extraversion: "Extraversion",
  agreeableness: "Agreeableness",
  neuroticism: "Neuroticism",
};

const elements = {
  scenarioControl: document.querySelector("#scenario-control"),
  conditionControl: document.querySelector("#condition-control"),
  personaList: document.querySelector("#persona-list"),
  quote: document.querySelector("#qualitative-response"),
  responseControls: document.querySelector("#response-controls"),
  nextResponse: document.querySelector("#next-response"),
  sprite: document.querySelector("#character-sprite"),
  personaCode: document.querySelector("#persona-code"),
  personaCaption: document.querySelector("#persona-caption"),
  oceanStats: document.querySelector("#ocean-stats"),
  sampleSize: document.querySelector("#sample-size"),
  experienceScores: document.querySelector("#experience-scores"),
  contextScenario: document.querySelector("#context-scenario"),
  contextCondition: document.querySelector("#context-condition"),
};

const state = {
  data: null,
  persona: null,
  scenario: null,
  condition: null,
  response: 0,
};

function byId(rows, id) {
  return rows.find((row) => row.id === id);
}

function resultKey(persona, scenario, condition) {
  return `${persona}|${scenario}|${condition}`;
}

function validateData(data) {
  if (!data || !Array.isArray(data.personas) || !Array.isArray(data.results)) {
    throw new Error("The persona results file does not match the explorer contract.");
  }
  const expected = new Set();
  for (const persona of data.personas) {
    for (const scenario of data.scenarios) {
      for (const condition of data.conditions) {
        expected.add(resultKey(persona.id, scenario.id, condition.id));
      }
    }
  }
  const observed = new Set();
  for (const row of data.results) {
    const key = resultKey(row.persona, row.scenario, row.condition);
    if (observed.has(key)) throw new Error(`Duplicate persona result: ${key}`);
    observed.add(key);
    for (const score of Object.keys(SCORE_LABELS)) {
      const value = Number(row.scores?.[score]);
      if (!Number.isFinite(value) || value < 1 || value > 7) {
        throw new Error(`Invalid ${score} score for ${key}`);
      }
    }
    if (
      !Array.isArray(row.quotes) ||
      row.quotes.length < 1 ||
      row.quotes.length > 3 ||
      row.quotes.some((quote) => typeof quote !== "string" || !quote.trim())
    ) {
      throw new Error(`Invalid qualitative responses for ${key}`);
    }
  }
  const missing = [...expected].filter((key) => !observed.has(key));
  if (missing.length) throw new Error(`Missing persona results: ${missing.join(", ")}`);
}

function selectedResult() {
  return state.data.results.find(
    (row) =>
      row.persona === state.persona &&
      row.scenario === state.scenario &&
      row.condition === state.condition,
  );
}

function responseOptions(result) {
  return result.quotes;
}

function portraitPath(sprite) {
  return sprite.replace(/\.png$/u, "-portrait.png");
}

function updateUrl() {
  const url = new URL(window.location.href);
  url.searchParams.set("persona", state.persona);
  url.searchParams.set("scenario", state.scenario);
  url.searchParams.set("condition", state.condition);
  url.searchParams.set("response", String(state.response));
  window.history.replaceState({}, "", url);
}

function makeSegment(container, rows, selected, onSelect) {
  container.replaceChildren();
  for (const row of rows) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.value = row.id;
    button.textContent = row.label;
    button.setAttribute("aria-pressed", String(row.id === selected));
    button.addEventListener("click", () => onSelect(row.id));
    container.append(button);
  }
}

function renderPersonaRail() {
  elements.personaList.replaceChildren();
  state.data.personas.forEach((persona, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "persona-button";
    button.dataset.persona = persona.id;
    button.style.setProperty("--persona-accent", persona.accent);
    button.style.setProperty("--persona-soft", persona.accent_soft);
    button.setAttribute("aria-label", `Select ${persona.name}`);
    button.setAttribute("aria-pressed", String(persona.id === state.persona));

    const image = document.createElement("img");
    image.src = portraitPath(persona.sprite);
    image.alt = "";
    image.width = 128;
    image.height = 128;
    button.append(image);
    button.addEventListener("click", () => {
      state.persona = persona.id;
      state.response = 0;
      render();
    });
    elements.personaList.append(button);
    if (persona.id === state.persona) elements.personaCode.textContent = String(index + 1).padStart(2, "0");
  });
}

function makeMeterRow(label, value, maximum, className) {
  const row = document.createElement("div");
  row.className = className;

  const name = document.createElement("span");
  name.className = className === "stat-row" ? "stat-name" : "score-name";
  if (className === "stat-row") {
    const initial = document.createElement("span");
    initial.className = "ocean-initial";
    initial.textContent = label[0];
    name.append(initial, label.slice(1));
  } else {
    name.textContent = label;
  }

  const numeric = document.createElement("span");
  numeric.className = className === "stat-row" ? "stat-value" : "score-value";
  numeric.textContent = className === "stat-row" ? String(value) : Number(value).toFixed(1);

  const meter = document.createElement("span");
  meter.className = "meter";
  meter.setAttribute("aria-hidden", "true");
  const fill = document.createElement("span");
  fill.className = "meter-fill";
  fill.style.width = `${Math.max(0, Math.min(100, (Number(value) / maximum) * 100))}%`;
  meter.append(fill);

  if (className === "stat-row") row.append(name, meter, numeric);
  else row.append(name, numeric, meter);
  return row;
}

function render() {
  const persona = byId(state.data.personas, state.persona);
  const scenario = byId(state.data.scenarios, state.scenario);
  const condition = byId(state.data.conditions, state.condition);
  const result = selectedResult();
  if (!persona || !scenario || !condition || !result) throw new Error("Invalid explorer selection.");

  document.documentElement.style.setProperty("--accent", persona.accent);
  document.documentElement.style.setProperty("--accent-dark", persona.accent_dark);
  document.documentElement.style.setProperty("--accent-soft", persona.accent_soft);

  makeSegment(elements.scenarioControl, state.data.scenarios, state.scenario, (id) => {
    state.scenario = id;
    state.response = 0;
    render();
  });
  makeSegment(elements.conditionControl, state.data.conditions, state.condition, (id) => {
    state.condition = id;
    state.response = 0;
    render();
  });
  renderPersonaRail();

  elements.personaCaption.textContent = persona.name;
  elements.sprite.src = persona.sprite;
  elements.sprite.alt = `${persona.name} pixel-art character`;
  elements.sprite.dataset.persona = persona.id;

  elements.oceanStats.replaceChildren(
    ...Object.entries(OCEAN_LABELS).map(([key, label]) =>
      makeMeterRow(label, persona.ocean[key], 100, "stat-row"),
    ),
  );
  elements.experienceScores.replaceChildren(
    ...Object.entries(SCORE_LABELS).map(([key, label]) =>
      makeMeterRow(label, result.scores[key], 7, "score-row"),
    ),
  );
  elements.sampleSize.textContent = result.sample_size ? `n = ${result.sample_size}` : "n pending";
  elements.contextScenario.textContent = scenario.label;
  elements.contextCondition.textContent = condition.label;

  const responses = responseOptions(result);
  state.response = ((state.response % responses.length) + responses.length) % responses.length;
  elements.quote.textContent = responses[state.response];
  elements.responseControls.hidden = responses.length < 2;
  elements.nextResponse.onclick = () => {
    state.response = (state.response + 1) % responses.length;
    elements.quote.textContent = responses[state.response];
    updateUrl();
  };
  updateUrl();
}

async function start() {
  const response = await fetch(DATA_URL, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load ${DATA_URL}`);
  const data = await response.json();
  validateData(data);
  state.data = data;

  const params = new URLSearchParams(window.location.search);
  const requestedPersona = params.get("persona");
  const requestedScenario = params.get("scenario");
  const requestedCondition = params.get("condition");
  state.persona = byId(data.personas, requestedPersona)?.id ?? data.personas[0].id;
  state.scenario = byId(data.scenarios, requestedScenario)?.id ?? data.scenarios[0].id;
  state.condition = byId(data.conditions, requestedCondition)?.id ?? data.conditions[0].id;
  state.response = Math.max(0, Number.parseInt(params.get("response") ?? "0", 10) || 0);
  document.body.classList.toggle("paper-mode", params.get("paper") === "1");
  document.body.dataset.resultStatus = data.meta?.status ?? "unknown";
  render();
}

start().catch((error) => {
  document.body.className = "load-error";
  document.body.textContent = `Persona explorer could not start: ${error.message}`;
  console.error(error);
});
