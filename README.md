# Minnesota US Senate Democratic Primary — Live Model

County-level Bayesian live election-night model for the 2026 Minnesota US Senate
Democratic Primary (Angie Craig vs. Peggy Flanagan), fed by civicAPI race `85562`.

Results from [civicAPI](https://civicapi.org).

## How it fits together

```
civicAPI  ──►  Render web service  ──►  Cloudflare Pages
 (poll)         (model + JSON API)        (the site)
```

One backend service. It polls civicAPI on a background thread, runs the model, and
serves the result over HTTP. The site reads that URL directly.

**This is a web service, not a cron job.** A cron container is destroyed after every
run, which wipes the turnout-calibration and shift state the model accumulates over
the night, and it has no URL for a site to read. The web service solves both by
staying alive.

## Counted votes are held fixed -- what's "not deductive" is narrower than that

The Michigan/Wisconsin/South Dakota family of models holds a county's counted votes
as literal fixed truth and only projects the uncounted remainder:
`projected_votes = counted_votes + remaining_votes * rate`. This model does that
too -- counted votes are never revised, full stop. What's different is only how
`rate` gets picked for the remainder: instead of trusting the county's own observed
rate alone (weighted by completeness), it's a credibility-weighted **blend** of the
county's own observed rate and a (shift-adjusted) baseline rate, where credibility
grows with how much of the county has actually reported. A small early batch that
disagrees with the baseline pulls the RATE FOR THE REMAINDER only partway toward
itself, not all the way -- it can never pull down a vote that's already been
counted.

The escape hatch is the statewide/regional shift layer: if **many** counties —
not just one, however large — show the same directional surprise against their
baselines, the shift converges toward that surprise's true size and gets folded
into every county's baseline before the per-county blend even happens. A single
large county partially reporting (Hennepin, say) is explicitly capped so it can't
read as a "consistent pattern" on its own — see `MAX_SINGLE_COUNTY_SHARE` in
`minnesota_senate_model.py`.

Full reasoning, with the formulas, is in `minnesota_senate_model.py`'s module
docstring.

## Files

| File | Does |
|---|---|
| `server.py` | background poller + JSON API. The entrypoint |
| `civicapi_feed.py` | API client, payload parsing, county name matching |
| `minnesota_senate_model.py` | baseline loading, credibility blending, shift shrinkage, turnout recalibration, Monte Carlo |
| `mn_county_baseline.csv` | the 87-county baseline the model loads at startup |
| `mn-counties.geojson` / `web/mn-counties.geojson` | county shapes for the map |
| `web/` | the static site (`index.html`, `app.js`, `style.css`) |

## Endpoints

| Route | Returns |
|---|---|
| `/health` | uptime, cycle count, last error |
| `/api/projection` | the current projection, county table, diagnostics |
| `/api/history` | one compact record per cycle since start |

CORS is open, so the site can be hosted anywhere.

## One cycle

```
fetch civicAPI
  → fold into per-county vote counts
  → recalibrate turnout from percent reporting
  → recompute statewide + regional shift (evidence-weighted, outlier-dampened,
    single-county-capped)
  → blend every county's own results against its shift-adjusted baseline
  → Monte Carlo, 20,000 sims
  → publish
```

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `RACE_ID` | civicAPI race | `85562` |
| `N_SIMS` | Monte Carlo draws | `20000` |
| `POLL_INTERVAL` | seconds between cycles | `60` |
| `STATE_DIR` | optional disk path so turnout/shift state survives a restart | unset |

## Known limitations

- **civicAPI's payload for this race is UNVERIFIED.** Built from the same schema
  pattern that's held for MI, SD, GA, and WI so far, but never confirmed against a
  real MN response. Get a sample early and fix `civicapi_feed.py` calmly rather
  than during live counting.
- **`percent_reporting` counts precincts, not votes**, same caution as every prior
  build in this family. Minnesota doesn't separate absentee from Election Day in
  its official results at all (unlike Michigan), so there's no mode-inference layer
  here — turnout calibration handles the precinct-vs-vote gap; there's no
  equivalent gap to close on vote *method*.
- **Evidence-prior constants (`GLOBAL_EVIDENCE_PRIOR`, `REGIONAL_EVIDENCE_PRIOR`,
  `MAX_SINGLE_COUNTY_SHARE`) are tuned against synthetic scenarios**, not a real
  count. They're vote-count-scaled (not the small WI-style constants), calibrated
  so one large county alone barely moves the shift while a genuine ~40+ county
  pattern converges strongly by 70% reporting. Watch the diagnostics on election
  night and retune if the shift looks too sticky or too twitchy.
- **The baseline itself** (`mn_county_baseline.csv`) comes from a pixel-sampled
  read of a crowd-prediction map plus a 2020 Sanders/Warren coalition-shape
  covariate — see `minnesota-governor-model.md`-style notes for that build's own
  caveats. It's a real signal, not a placeholder, but it's still a baseline.
- **Margins are two-candidate.** Any other filed candidate is dropped from the
  denominator entirely by `civicapi_feed.py`, matching how the baseline was built.
- **State is in memory.** A restart costs the shift/turnout calibration until
  counties report again. Set `STATE_DIR` to a mounted disk to avoid that.
