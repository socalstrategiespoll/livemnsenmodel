// ---------------------------------------------------------------------------
// Point this at your Render service URL. No trailing slash.
// ---------------------------------------------------------------------------
const API_BASE = "https://mn-senate-primary-model.onrender.com";

const REFRESH_MS = 15000;
const STALE_AFTER_MS = 180000;

const FLANAGAN = "Flanagan";
const CRAIG = "Craig";

const num = new Intl.NumberFormat("en-US");
const $ = (id) => document.getElementById(id);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function signed(v, d = 1) {
  return (v >= 0 ? "+" : "\u2212") + Math.abs(v).toFixed(d);
}

// ---------------------------------------------------------------------------
// Distribution -- drawn from the simulated percentiles the model produced.
// ---------------------------------------------------------------------------

const W = 720;
const H = 200;
const PAD = 8;

function density(percentiles, bins = 48) {
  const lo = percentiles[0];
  const hi = percentiles[percentiles.length - 1];
  const span = hi - lo || 1;
  const counts = new Array(bins).fill(0);

  percentiles.forEach((v) => {
    const i = Math.min(bins - 1, Math.floor(((v - lo) / span) * bins));
    counts[i] += 1;
  });

  const smooth = counts.map((_, i) => {
    const a = counts[i - 1] ?? counts[i];
    const b = counts[i];
    const c = counts[i + 1] ?? counts[i];
    return (a + 2 * b + c) / 4;
  });

  const peak = Math.max(...smooth, 1);
  return { lo, hi, span, values: smooth.map((v) => v / peak) };
}

function curvePath(d, close) {
  const step = W / (d.values.length - 1);
  const y = (v) => PAD + (1 - v) * (H - 2 * PAD - 24);
  let path = `M 0 ${y(d.values[0]).toFixed(1)}`;

  for (let i = 1; i < d.values.length; i++) {
    const x0 = (i - 1) * step;
    const x1 = i * step;
    const mid = (x0 + x1) / 2;
    path += ` C ${mid.toFixed(1)} ${y(d.values[i - 1]).toFixed(1)},` +
            ` ${mid.toFixed(1)} ${y(d.values[i]).toFixed(1)},` +
            ` ${x1.toFixed(1)} ${y(d.values[i]).toFixed(1)}`;
  }

  if (close) path += ` L ${W} ${H - 24} L 0 ${H - 24} Z`;
  return path;
}

function drawDistribution(p) {
  const pct = p.margin_percentiles;
  if (!pct || pct.length < 8) return;

  const d = density(pct);
  const xOf = (v) => ((v - d.lo) / d.span) * W;
  const clamp = (x) => Math.max(0, Math.min(W, x));

  $("dist-fill").setAttribute("d", curvePath(d, true));
  $("dist-line").setAttribute("d", curvePath(d, false));

  const band = $("dist-band");
  band.innerHTML = "";
  const rect = (x1, x2, cls) => {
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    r.setAttribute("x", clamp(x1));
    r.setAttribute("y", 4);
    r.setAttribute("width", Math.max(0, clamp(x2) - clamp(x1)));
    r.setAttribute("height", H - 28);
    r.setAttribute("class", cls);
    band.appendChild(r);
  };
  rect(xOf(p.interval_90[0]), xOf(p.interval_90[1]), "band-rect-90");
  rect(xOf(p.interval_50[0]), xOf(p.interval_50[1]), "band-rect-50");

  const zero = $("dist-zero");
  const zeroX = clamp(xOf(0));
  zero.setAttribute("x1", zeroX);
  zero.setAttribute("x2", zeroX);
  zero.style.display = (0 >= d.lo && 0 <= d.hi) ? "" : "none";

  const med = $("dist-median");
  const medX = clamp(xOf(p.median_margin));
  med.setAttribute("x1", medX);
  med.setAttribute("x2", medX);

  $("dist-axis").innerHTML =
    `<span>${signed(d.lo)}</span>` +
    `<span>${p.median_margin >= 0 ? FLANAGAN : CRAIG} margin</span>` +
    `<span>${signed(d.hi)}</span>`;

  // Numeric ranges on the key itself, not just the color swatches -- the
  // middle 50% and middle 90% of simulated outcomes, as actual numbers.
  $("dist-key").innerHTML =
    `<span><i class="swatch band-90"></i>Middle 90%: ${signed(p.interval_90[0])} to ${signed(p.interval_90[1])}</span>` +
    `<span><i class="swatch band-50"></i>Middle 50%: ${signed(p.interval_50[0])} to ${signed(p.interval_50[1])}</span>` +
    `<span><i class="swatch tick-zero"></i>tie</span>`;
}


// ---------------------------------------------------------------------------
// Maps
//
// Real GeoJSON in lat/lon, projected in-browser with an Albers conic tuned to
// Minnesota (roughly 43.5N-49.4N, -97.2 to -89.5).
//
// Counted votes are always held fixed here too -- what varies between the two
// maps is only how the UNCOUNTED remainder gets rated. COUNTED shows the raw,
// unmodeled margin in whatever's actually been tallied so far (grey where
// nothing has); MODEL PROJECTION shows counted votes plus the remainder
// projected at a credibility-weighted blended rate (see
// minnesota_senate_model.py's project_margin()), which exists for every
// county from the first cycle on, since the remainder starts from the
// pre-election baseline before any votes are in.
// ---------------------------------------------------------------------------

const MAP_W = 620;
const MAP_H = 700;
const MAP_SCALE = 25;

const MAPS = [
  { id: "counted", label: "Counted so far",
    note: "Raw margin in the votes already reported. Grey has not reported." },
  { id: "projected", label: "Model projection",
    note: "The model's blended projection -- baseline and observed results combined, not just the raw count." },
];

let GEO = null;
let PATHS = {};
let LAST_COUNTIES = [];
let PINNED = null;
let VIEW = { x: 0, y: 0, k: 1 };

// --- projection: Albers conic tuned to Minnesota ----------------------------

function albers() {
  const LAT0 = 46.4, LON0 = -93.4, P1 = 44.5, P2 = 48.3;
  const rad = Math.PI / 180;
  const n = 0.5 * (Math.sin(P1 * rad) + Math.sin(P2 * rad));
  const C = Math.cos(P1 * rad) ** 2 + 2 * n * Math.sin(P1 * rad);
  const rho0 = Math.sqrt(C - 2 * n * Math.sin(LAT0 * rad)) / n;
  return ([lon, lat]) => {
    const theta = n * (lon - LON0) * rad;
    const rho = Math.sqrt(Math.max(C - 2 * n * Math.sin(lat * rad), 1e-12)) / n;
    return [rho * Math.sin(theta), rho0 - rho * Math.cos(theta)];
  };
}

function buildPaths(geo) {
  const project = albers();
  const projected = geo.features.map((f) => {
    // GeoJSON Polygon vs MultiPolygon need different coordinate nesting --
    // Minnesota's county file has both (most counties are plain Polygon; any
    // county straddling an island or exclave on Lake of the Woods/Superior
    // could be MultiPolygon), so both are normalized before rendering.
    const polys = f.geometry.type === "Polygon"
      ? [f.geometry.coordinates]
      : f.geometry.coordinates;
    return {
      name: f.properties.name,
      polys: polys.map((poly) => poly.map((r) => r.map(project))),
    };
  });

  let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
  projected.forEach((f) => f.polys.forEach((poly) => poly.forEach((r) => r.forEach(([x, y]) => {
    if (x < minx) minx = x;
    if (x > maxx) maxx = x;
    if (y < miny) miny = y;
    if (y > maxy) maxy = y;
  }))));

  const pad = 8;
  const k = Math.min((MAP_W - 2 * pad) / (maxx - minx), (MAP_H - 2 * pad) / (maxy - miny));
  const ox = (MAP_W - (maxx - minx) * k) / 2;
  const oy = (MAP_H - (maxy - miny) * k) / 2;

  const out = {};
  projected.forEach((f) => {
    let d = "";
    f.polys.forEach((poly) => poly.forEach((r) => {
      if (r.length < 4) return;
      d += "M" + r.map(([x, y]) =>
        ((x - minx) * k + ox).toFixed(1) + " " + ((maxy - y) * k + oy).toFixed(1)
      ).join("L") + "Z";
    }));
    out[f.name] = d;
  });
  return out;
}

// --- color --------------------------------------------------------------

function rampColor(margin) {
  if (margin === null || margin === undefined) return "#C7CCC2";
  const t = Math.max(-1, Math.min(1, margin / MAP_SCALE));
  const mid = [232, 234, 227];
  const end = t >= 0 ? [31, 111, 107] : [150, 112, 26];
  const k = Math.pow(Math.abs(t), 0.75);
  const c = mid.map((m, i) => Math.round(m + (end[i] - m) * k));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function countyRow(name) {
  return LAST_COUNTIES.find((r) => r.county === name);
}

function mapValue(row, mode) {
  if (!row) return null;
  if (mode === "counted") return row.reporting ? row.margin : null;
  return row.projected_final;
}

// --- build ----------------------------------------------------------------

async function loadGeo() {
  try {
    const res = await fetch("mn-counties.geojson", { cache: "no-store" });
    GEO = await res.json();
    PATHS = buildPaths(GEO);
    MAPS.forEach(buildOne);
    paintMaps();
  } catch (err) {
    const el = document.querySelector(".maps");
    if (el) el.hidden = true;
  }
}

function buildOne(map) {
  const g = document.getElementById("shapes-" + map.id);
  g.innerHTML = "";
  Object.entries(PATHS).forEach(([county, d]) => {
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", d);
    p.setAttribute("fill", "#C7CCC2");
    p.dataset.county = county;
    p.addEventListener("pointerenter", () => hover(county, map.id));
    p.addEventListener("pointerleave", () => hover(null, map.id));
    p.addEventListener("click", (e) => {
      e.stopPropagation();
      PINNED = PINNED === county ? null : county;
      paintDetail(PINNED || county);
      markSelected();
    });
    g.appendChild(p);
  });
  document.getElementById("note-" + map.id).textContent = map.note;
}

function paintMaps() {
  if (!GEO) return;
  MAPS.forEach((map) => {
    document.getElementById("shapes-" + map.id)
      .querySelectorAll("path").forEach((p) => {
        p.setAttribute("fill", rampColor(mapValue(countyRow(p.dataset.county), map.id)));
      });
    const empty = map.id === "counted" ? "no results yet" : "no baseline";
    document.getElementById("legend-" + map.id).innerHTML =
      `<div class="legend-scale">${[-1, -0.6, -0.3, 0, 0.3, 0.6, 1]
        .map((t) => `<span style="background:${rampColor(t * MAP_SCALE)}"></span>`).join("")}</div>
       <div class="legend-ends"><span>Craig +${MAP_SCALE}</span><span>tie</span><span>Flanagan +${MAP_SCALE}</span></div>
       <div class="legend-none"><i></i>${empty}</div>`;
  });
  markSelected();
  paintDetail(PINNED || $("map-detail").dataset.county || null);
}

function markSelected() {
  MAPS.forEach((map) => {
    document.getElementById("shapes-" + map.id)
      .querySelectorAll("path").forEach((p) => {
        p.classList.toggle("sel", p.dataset.county === PINNED);
        p.classList.toggle("hov", p.dataset.county === HOVER);
      });
  });
  $("pin-note").hidden = !PINNED;
}

let HOVER = null;

function hover(county, _from) {
  HOVER = county;
  markSelected();
  if (county) paintDetail(county);
  else if (PINNED) paintDetail(PINNED);
}

// --- detail panel -----------------------------------------------------------

function paintDetail(county) {
  const box = $("map-detail");
  const r = county && countyRow(county);
  if (!r) {
    box.innerHTML = `<p class="map-hint">Hover a county on either map. Click to pin it.</p>`;
    box.dataset.county = "";
    return;
  }
  box.dataset.county = county;
  const cls = (v) => (v >= 0 ? "v-flanagan" : "v-craig");

  box.innerHTML = `
    <h3>${county}</h3>
    <dl>
      <dt>Region</dt><dd>${r.region}</dd>
      <dt>Counted</dt><dd>${num.format(r.votes)} of ${num.format(r.projected_total)}</dd>
      <dt>Margin so far</dt>
      <dd class="${r.margin === null ? "" : cls(r.margin)}">${r.margin === null ? "—" : signed(r.margin)}</dd>
      <dt>Pre-election baseline</dt><dd>${signed(r.expected_baseline)}</dd>
      <dt>Credibility given to own results</dt><dd>${(r.credibility * 100).toFixed(0)}%</dd>
      <dt>Model projection</dt>
      <dd class="${cls(r.projected_final)}">${signed(r.projected_final)}</dd>
    </dl>
    <p class="split-note">
      ${r.calibrated_turnout && r.calibrated_turnout !== r.projected_total
        ? `Turnout here is now projected at ${num.format(r.calibrated_turnout)}, revised from the pre-election baseline based on this county's own reporting pace.`
        : `Turnout still running on the pre-election baseline projection.`}
      Counted votes are always held fixed. Only the uncounted remainder is
      modeled -- blended between this county's own results so far and the
      baseline at a weight (credibility) that grows with how much of the
      county has reported, and shifts further toward observed results if
      many other counties are showing the same pattern.
    </p>`;
}

// --- zoom and pan -----------------------------------------------------------

function applyView() {
  const t = `translate(${VIEW.x} ${VIEW.y}) scale(${VIEW.k})`;
  MAPS.forEach((m) => {
    document.getElementById("shapes-" + m.id).setAttribute("transform", t);
    document.getElementById("shapes-" + m.id)
      .style.setProperty("--stroke", (0.6 / VIEW.k).toFixed(2) + "px");
  });
  $("zoom-reset").hidden = VIEW.k === 1 && VIEW.x === 0 && VIEW.y === 0;
}

function zoomAt(svg, factor, cx, cy) {
  const box = svg.getBoundingClientRect();
  const px = ((cx - box.left) / box.width) * MAP_W;
  const py = ((cy - box.top) / box.height) * MAP_H;
  const k = Math.max(1, Math.min(12, VIEW.k * factor));
  VIEW.x = px - ((px - VIEW.x) / VIEW.k) * k;
  VIEW.y = py - ((py - VIEW.y) / VIEW.k) * k;
  VIEW.k = k;
  clampView();
  applyView();
}

function clampView() {
  const limX = MAP_W * (VIEW.k - 1);
  const limY = MAP_H * (VIEW.k - 1);
  VIEW.x = Math.max(-limX, Math.min(0, VIEW.x));
  VIEW.y = Math.max(-limY, Math.min(0, VIEW.y));
}

function initMapInteraction() {
  MAPS.forEach((map) => {
    const svg = document.getElementById("svg-" + map.id);

    svg.addEventListener("wheel", (e) => {
      e.preventDefault();
      zoomAt(svg, e.deltaY < 0 ? 1.18 : 1 / 1.18, e.clientX, e.clientY);
    }, { passive: false });

    let drag = null;
    svg.addEventListener("pointerdown", (e) => {
      if (VIEW.k === 1) return;
      drag = { x: e.clientX, y: e.clientY, vx: VIEW.x, vy: VIEW.y };
      svg.setPointerCapture(e.pointerId);
      svg.classList.add("dragging");
    });
    svg.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const box = svg.getBoundingClientRect();
      VIEW.x = drag.vx + ((e.clientX - drag.x) / box.width) * MAP_W;
      VIEW.y = drag.vy + ((e.clientY - drag.y) / box.height) * MAP_H;
      clampView();
      applyView();
    });
    const stop = (e) => {
      if (!drag) return;
      drag = null;
      svg.releasePointerCapture(e.pointerId);
      svg.classList.remove("dragging");
    };
    svg.addEventListener("pointerup", stop);
    svg.addEventListener("pointercancel", stop);
  });

  $("zoom-in").addEventListener("click", () => {
    const svg = document.getElementById("svg-counted");
    const b = svg.getBoundingClientRect();
    zoomAt(svg, 1.5, b.left + b.width / 2, b.top + b.height / 2);
  });
  $("zoom-out").addEventListener("click", () => {
    const svg = document.getElementById("svg-counted");
    const b = svg.getBoundingClientRect();
    zoomAt(svg, 1 / 1.5, b.left + b.width / 2, b.top + b.height / 2);
  });
  $("zoom-reset").addEventListener("click", () => {
    VIEW = { x: 0, y: 0, k: 1 };
    applyView();
  });
  $("unpin").addEventListener("click", () => {
    PINNED = null;
    markSelected();
    paintDetail(null);
  });
}

// ---------------------------------------------------------------------------

let lastMargin = null;

function animateMargin(el, to) {
  const from = lastMargin;
  lastMargin = to;
  if (reduceMotion || from === null || Math.abs(to - from) < 0.05) {
    el.textContent = signed(to);
    return;
  }
  const start = performance.now();
  const dur = 550;
  const tick = (now) => {
    const t = Math.min(1, (now - start) / dur);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = signed(from + (to - from) * eased);
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function renderCounties(all) {
  const rows = (all || []).filter((r) => r.reporting);
  const body = $("county-rows");
  if (!rows.length) {
    body.innerHTML = `<tr class="empty"><td colspan="7">No counties reporting yet.</td></tr>`;
    return;
  }
  body.innerHTML = rows.map((r) => {
    const cls = r.margin >= 0 ? "v-flanagan" : "v-craig";
    const sw = r.vs_expected >= 0 ? "v-flanagan" : "v-craig";
    return `<tr>
      <td class="name">${r.county}</td>
      <td class="num">${r.pct_of_projected.toFixed(0)}%</td>
      <td class="num">${num.format(r.flanagan)}</td>
      <td class="num">${num.format(r.craig)}</td>
      <td class="num ${cls}">${signed(r.margin)}</td>
      <td class="num">${signed(r.expected_baseline)}</td>
      <td class="num ${sw}">${signed(r.vs_expected)}</td>
    </tr>`;
  }).join("");
}

function renderRegions(shifts) {
  const wrap = $("regions");
  const entries = Object.entries(shifts).sort((a, b) => b[1] - a[1]);
  const max = Math.max(2, ...entries.map(([, v]) => Math.abs(v)));

  wrap.innerHTML = entries.map(([name, v]) => {
    const half = (Math.abs(v) / max) * 50;
    const pos = v >= 0;
    return `<div class="region">
      <div>
        <div class="region-name">${name}</div>
        <div class="region-track">
          <span class="region-mid"></span>
          <span class="region-fill ${pos ? "pos" : "neg"}"
                style="left:${pos ? 50 : 50 - half}%;width:${half}%"></span>
        </div>
      </div>
      <div class="region-val ${pos ? "v-flanagan" : "v-craig"}">${signed(v, 2)}</div>
    </div>`;
  }).join("");
}

function render(data) {
  const p = data.projection;
  const c = data.counted;
  const d = data.diagnostics;
  const t = data.turnout;

  const leadFlanagan = p.median_margin >= 0;
  const leader = leadFlanagan ? FLANAGAN : CRAIG;
  const prob = leadFlanagan ? p.flanagan_win_probability : p.craig_win_probability;

  $("lead-name").textContent = leader;
  $("lead-name").className = "verdict-name " + (leadFlanagan ? "flanagan" : "craig");
  $("lead-margin").className = "verdict-number " + (leadFlanagan ? "flanagan" : "craig");
  animateMargin($("lead-margin"), Math.abs(p.median_margin));

  $("verdict-sub").textContent = d.counties_reporting === 0
    ? "Pre-election baseline. No counties reporting."
    : `From ${d.counties_reporting} ${d.counties_reporting === 1 ? "county" : "counties"}` +
      ` and ${c.pct_of_projected_turnout.toFixed(1)}% of the projected vote.`;

  drawDistribution(p);

  $("win-prob").textContent = (prob * 100).toFixed(prob > 0.995 ? 1 : 0) + "%";
  $("win-note").textContent = `${leader} wins in ${(prob * 100).toFixed(1)}% of simulations`;

  $("counted").textContent = c.pct_of_projected_turnout.toFixed(1) + "%";
  $("precincts").textContent =
    c.pct_precincts_reporting == null ? "—" : c.pct_precincts_reporting + "%";

  $("turnout").textContent = t.projected ? num.format(t.projected) : "—";

  const flanaganShare = p.flanagan_pct;
  const craigShare = p.craig_pct;
  $("flanagan-votes").textContent = num.format(c.flanagan || 0);
  $("craig-votes").textContent = num.format(c.craig || 0);
  $("tally-flanagan").style.width = flanaganShare + "%";
  $("tally-craig").style.width = craigShare + "%";
  $("flanagan-pct").textContent = flanaganShare.toFixed(1) + "%";
  $("craig-pct").textContent = craigShare.toFixed(1) + "%";

  LAST_COUNTIES = data.counties || [];
  renderCounties(data.counties);
  paintMaps();
  renderRegions(data.regional_shift || {});

  $("d-counties").textContent = d.counties_reporting;
  $("d-shift").textContent = signed(d.statewide_shift, 2);
  $("d-ci50").textContent = `${signed(p.interval_50[0])} to ${signed(p.interval_50[1])}`;
  $("d-ci90").textContent = `${signed(p.interval_90[0])} to ${signed(p.interval_90[1])}`;

  const stamp = new Date(data.updated_at);
  const stale = Date.now() - stamp.getTime() > STALE_AFTER_MS;
  setPulse(stale ? "stale" : "live", stale ? "feed stale" : "live", stamp);

  if (d.unmatched_counties && d.unmatched_counties.length) {
    $("alert").textContent =
      "Not matched to a model county, and excluded from the projection: " +
      d.unmatched_counties.join(", ");
    $("alert").hidden = false;
  } else {
    $("alert").hidden = true;
  }
}

function setPulse(state, label, stamp) {
  $("pulse").dataset.state = state;
  $("pulse-label").textContent = label;
  if (stamp) {
    $("stamp").textContent = stamp.toLocaleTimeString([], {
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  }
}

async function tick() {
  try {
    const res = await fetch(API_BASE + "/api/projection", { cache: "no-store" });
    if (res.status === 503) {
      setPulse("connecting", "waiting for first results");
      return;
    }
    if (!res.ok) throw new Error("HTTP " + res.status);
    render(await res.json());
  } catch (err) {
    setPulse("stale", "reconnecting");
  }
}

initMapInteraction();
loadGeo();
tick();
setInterval(tick, REFRESH_MS);
