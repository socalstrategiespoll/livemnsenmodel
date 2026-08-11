# Render web service: polls civicAPI on a background thread and serves the projection.
#
# Same reasoning as MI/WI for web service vs. cron job: a cron container is
# destroyed after every run, wiping the turnout-calibration and shift state
# this model accumulates over the night, and it has no URL for a site to read.
#
# WHAT'S DIFFERENT FROM THE WISCONSIN SERVER
#
#     The model itself is NOT deductive (see minnesota_senate_model.py's module
#     docstring for the full reasoning) -- every county's full projected result
#     comes from project_margin()'s credibility-weighted blend of its own
#     observed margin against a shift-adjusted baseline, not "counted votes held
#     fixed + blended remainder." Structurally that means build_output() reads
#     proj["flanagan_votes"]/["craig_votes"] (already the full blended
#     projection) rather than summing counted-plus-remainder separately.
#
#     Two candidates only, no "Other" bucket.
#
# DESIGN NOTES (unchanged from MI/WI)
#
#     Stdlib only, single-process threading server -- gunicorn with multiple
#     workers would spawn multiple pollers fighting over the API.
#
#     The poller never lets an exception escape. A civicAPI hiccup costs one
#     update; the previous projection stays served with its own timestamp.
#
#     State is in memory. Set STATE_DIR to a mounted Render disk if you want
#     turnout-calibration and shift history to survive a restart.

import json
import os
import threading
import time
import traceback

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from minnesota_senate_model import MinnesotaSenateModel, REGIONS
from civicapi_feed import fetch_race, parse_payload, MINNESOTA_SENATE_DEM_PRIMARY


PORT = int(os.environ.get("PORT", 10000))
RACE_ID = int(os.environ.get("RACE_ID", MINNESOTA_SENATE_DEM_PRIMARY))
N_SIMS = int(os.environ.get("N_SIMS", 20000))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", 60))
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", 2000))
STATE_DIR = os.environ.get("STATE_DIR", "")
BASELINE_PATH = os.environ.get("BASELINE_PATH", "mn_county_baseline.csv")


class ModelState:
    """Everything the poller produces and the HTTP handler reads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.projection = None
        self.history = []
        self.error = None
        self.cycles = 0
        self.started_at = datetime.now(timezone.utc).isoformat()

    def publish(self, output: dict) -> None:
        with self.lock:
            self.projection = output
            self.history.append({
                "updated_at": output["updated_at"],
                "flanagan_pct": output["projection"]["flanagan_pct"],
                "craig_pct": output["projection"]["craig_pct"],
                "flanagan_win_probability": output["projection"]["flanagan_win_probability"],
                "interval_90": output["projection"]["interval_90"],
                "pct_counted": output["counted"]["pct_of_projected_turnout"],
                "counties_reporting": output["diagnostics"]["counties_reporting"],
                "statewide_shift": output["diagnostics"]["statewide_shift"],
            })
            if len(self.history) > HISTORY_LIMIT:
                self.history = self.history[-HISTORY_LIMIT:]
            self.error = None
            self.cycles += 1

    def fail(self, message: str) -> None:
        with self.lock:
            self.error = message

    def snapshot(self) -> tuple:
        with self.lock:
            return self.projection, list(self.history), self.error, self.cycles


STATE = ModelState()


def build_output(model: MinnesotaSenateModel, sim: dict, proj: dict,
                 parsed: dict, race_id: int) -> dict:
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "civicapi.org",
        "attribution": "Election results from civicAPI (civicapi.org)",
        "race_id": race_id,
        "election_name": parsed.get("election_name"),
        "feed_last_updated": parsed.get("last_updated"),
        "counted": {
            "flanagan": parsed.get("state_flanagan"),
            "craig": parsed.get("state_craig"),
            "pct_of_projected_turnout": round(100 * proj["pct_counted"], 2),
            "pct_precincts_reporting": parsed.get("percent_precincts_statewide"),
        },
        "turnout": {
            "projected": round(proj["projected_turnout"]),
            "turnout_vs_prior": round(proj["turnout_pooled_ratio"], 3),
        },
        "projection": {
            "flanagan_win_probability": round(sim["flanagan_win_prob"], 4),
            "craig_win_probability": round(sim["craig_win_prob"], 4),
            "median_margin": round(sim["median_margin"], 2),
            "interval_50": [round(sim["p25"], 2), round(sim["p75"], 2)],
            "interval_90": [round(sim["p05"], 2), round(sim["p95"], 2)],
            "margin_percentiles": sim["margin_percentiles"],
            "flanagan_pct": round(proj["flanagan_pct"], 2),
            "craig_pct": round(proj["craig_pct"], 2),
            "flanagan_votes": int(proj["flanagan_votes"]),
            "craig_votes": int(proj["craig_votes"]),
        },
        "counties": build_county_table(model),
        "diagnostics": {
            "counties_reporting": proj["n_reported"],
            "statewide_shift": round(proj["statewide_shift"], 2),
            "total_evidence_weight": round(proj["total_evidence_weight"]),
            "unmatched_counties": parsed.get("unmatched", []),
            "candidate_names": parsed.get("candidate_names"),
        },
        "regional_shift": {r: round(v, 2) for r, v in proj["regional_shift"].items()},
    }


def build_county_table(model: MinnesotaSenateModel) -> list:
    """Per-county rows covering all 87 counties every cycle, not just the ones
    reporting -- the maps need every county.

    'margin' below is the RAW counted-so-far margin (honest, unmodeled) --
    what's actually been tallied. 'projected_final' is the model's blended
    projection for the county (see minnesota_senate_model.py), which can
    legitimately differ from the raw margin early, by design: this model
    doesn't treat a small early batch as gospel, it blends it against the
    baseline at a credibility weight that grows with how much of the county
    is actually in."""
    rows = []
    for name, c in model.counties.items():
        raw_margin = c.observed_margin
        projected_margin = model.project_margin(c)
        remaining = max(0, c.effective_turnout - c.counted_votes)

        rows.append({
            "county": name,
            "region": c.region,
            "reporting": c.counted_votes > 0,
            "flanagan": c.votes["flanagan"],
            "craig": c.votes["craig"],
            "votes": c.counted_votes,
            "margin": None if raw_margin is None else round(raw_margin, 1),
            "expected_baseline": round(c.baseline_margin, 1),
            "vs_expected": None if raw_margin is None else round(raw_margin - c.baseline_margin, 1),
            "credibility": round(c.credibility, 3),
            "first_batch": c.is_first_batch,
            "pct_precincts": round(c.pct_reporting * 100, 1) if c.pct_reporting else None,
            "pct_of_projected": round(100 * c.pct_counted, 1),
            "projected_total": int(c.effective_turnout),
            "calibrated_turnout": int(c.calibrated_turnout) if c.calibrated_turnout else None,
            "remaining": int(round(remaining)),
            "projected_final": round(projected_margin, 1),
        })

    rows.sort(key=lambda r: (-r["votes"], -r["projected_total"]))
    return rows


def save_state(model: MinnesotaSenateModel) -> None:
    if not STATE_DIR:
        return
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        snap = {
            name: {
                "flanagan": c.votes["flanagan"], "craig": c.votes["craig"],
                "pct_reporting": c.pct_reporting, "is_first_batch": c.is_first_batch,
            }
            for name, c in model.counties.items() if c.counted_votes > 0
        }
        with open(os.path.join(STATE_DIR, "feed_state.json"), "w") as handle:
            json.dump(snap, handle)
    except Exception:
        pass


def load_state(model: MinnesotaSenateModel) -> None:
    if not STATE_DIR:
        return
    path = os.path.join(STATE_DIR, "feed_state.json")
    try:
        with open(path) as handle:
            stored = json.load(handle)
        for name, rec in stored.items():
            if name in model.counties:
                model.update_county(name, rec["flanagan"], rec["craig"], rec["pct_reporting"])
                # update_county infers is_first_batch from the 0->nonzero
                # transition, which a restore skips (counted_votes starts at 0
                # either way) -- restore the actual saved flag afterward.
                model.counties[name].is_first_batch = rec.get("is_first_batch", False)
        print("restored {} counties from {}".format(len(stored), path), flush=True)
    except Exception:
        pass


def poller() -> None:
    """Background loop. Never exits."""
    model = MinnesotaSenateModel(BASELINE_PATH)
    load_state(model)
    county_names = list(model.counties.keys())

    print("poller started: race {} every {}s, {} sims".format(
        RACE_ID, POLL_INTERVAL, N_SIMS), flush=True)

    while True:
        started = time.time()
        try:
            payload = fetch_race(RACE_ID)
            parsed = parse_payload(payload, county_names)

            for county, record in parsed["counties"].items():
                model.update_county(county, record["flanagan"], record["craig"],
                                    record.get("percent_precincts"))

            sim = model.run_simulation(n_sims=N_SIMS)
            proj = model.project()
            output = build_output(model, sim, proj, parsed, RACE_ID)
            STATE.publish(output)
            save_state(model)

            names = output["diagnostics"].get("candidate_names") or {}
            if not names.get("flanagan") or not names.get("craig"):
                print("!! CANDIDATE MATCH FAILED: flanagan={!r} craig={!r} -- fix "
                      "FLANAGAN_KEYS / CRAIG_KEYS in civicapi_feed.py".format(
                          names.get("flanagan"), names.get("craig")), flush=True)
            else:
                print("   matched: {} vs {}".format(names["flanagan"], names["craig"]), flush=True)
            if output["diagnostics"]["unmatched_counties"]:
                print("!! UNMATCHED COUNTIES: {} -- fix normalize_county() in "
                      "civicapi_feed.py".format(
                          output["diagnostics"]["unmatched_counties"]), flush=True)

            p = output["projection"]
            print("[{}] {:.1f}% counted | {} cty | Flanagan {:.1f}  Craig {:.1f} | "
                  "margin {:+.1f} [{:+.1f}, {:+.1f}] | Flanagan win {:.1%}".format(
                      datetime.now().strftime("%H:%M:%S"),
                      output["counted"]["pct_of_projected_turnout"],
                      output["diagnostics"]["counties_reporting"],
                      p["flanagan_pct"], p["craig_pct"], p["median_margin"],
                      p["interval_90"][0], p["interval_90"][1],
                      p["flanagan_win_probability"]), flush=True)

        except Exception as exc:
            STATE.fail(str(exc))
            print("[{}] cycle failed, serving last good projection: {}".format(
                datetime.now().strftime("%H:%M:%S"), exc), flush=True)
            traceback.print_exc()

        time.sleep(max(1.0, POLL_INTERVAL - (time.time() - started)))


class Handler(BaseHTTPRequestHandler):

    def _send(self, body, status=200, content_type="application/json"):
        encoded = (body if isinstance(body, bytes) else json.dumps(body).encode("utf-8"))
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        projection, history, error, cycles = STATE.snapshot()

        if path in ("/", "/health"):
            return self._send({
                "ok": True, "cycles": cycles, "started_at": STATE.started_at,
                "last_error": error, "has_projection": projection is not None,
            })
        if path == "/api/projection":
            if projection is None:
                return self._send({"error": "no projection yet", "last_error": error}, status=503)
            return self._send(projection)
        if path == "/api/history":
            return self._send({"count": len(history), "cycles": history})
        return self._send({"error": "not found"}, status=404)

    def log_message(self, *args):
        return


def main():
    thread = threading.Thread(target=poller, daemon=True)
    thread.start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("serving on :{}".format(PORT), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
