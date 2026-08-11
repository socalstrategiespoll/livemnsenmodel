"""
civicAPI live feed for the Minnesota model.

Endpoint:  https://civicapi.org/api/v2/race/{race_id}
Race:      85562  (2026 Minnesota US Senate Democratic Primary)
Auth:      none. Attribution required for non-personal use, so credit civicapi.org
           anywhere this output is published.

Structurally identical to the Wisconsin client (same schema pattern verified
there): extracts Craig/Flanagan per county and hands them to
MinnesotaSenateModel.update_county(). percent_reporting is a PRECINCT metric,
not a vote-completeness metric -- same caution as every other build in this
family. UNVERIFIED against MN's actual payload; get a real sample before
trusting this on election night, same as every prior build in this family had
to before its own race.

Two-candidate only. Everyone else in the field (if any) is dropped, matching
the baseline's two-candidate convention.
"""

import re
import time
import unicodedata

try:
    import requests
except ImportError:
    requests = None

API_BASE = "https://civicapi.org/api/v2"
MINNESOTA_SENATE_DEM_PRIMARY = 85562

# Substring match keys -- VERIFY against the actual payload once reachable.
FLANAGAN_KEYS = ("flanagan",)
CRAIG_KEYS = ("craig",)

REQUEST_TIMEOUT = 15
MAX_RETRIES = 4


def normalize_county(name: str) -> str:
    """Reduce a county name to a matching key. Handles 'St. Louis' against
    'st_louis' or 'Saint Louis', 'Lac qui Parle' against odd casing/spacing,
    and a trailing 'County' if the feed adds one."""
    if name is None:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"\bcounty\b", " ", text)
    text = re.sub(r"\bsaint\b", "st", text)
    text = text.replace(".", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def build_county_lookup(county_names) -> dict:
    return {normalize_county(c): c for c in county_names}


def fetch_race(race_id: int = MINNESOTA_SENATE_DEM_PRIMARY,
               timeout: int = REQUEST_TIMEOUT,
               max_retries: int = MAX_RETRIES,
               session=None) -> dict:
    """GET a race payload, retrying on transient failure with backoff.
    Raises on exhaustion -- callers should catch and keep the last good snapshot."""
    if requests is None:
        raise RuntimeError("requests is not installed: pip install requests")

    url = "{}/race/{}".format(API_BASE, race_id)
    getter = session.get if session is not None else requests.get
    last_error = None

    for attempt in range(max_retries):
        try:
            response = getter(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError("civicAPI fetch failed after {} attempts: {}".format(
        max_retries, last_error))


def _match_candidate(name: str, keys: tuple) -> bool:
    lowered = str(name).lower()
    return any(k in lowered for k in keys)


def extract_two_candidate(candidate_list: list) -> tuple:
    """Pull Flanagan and Craig votes out of a candidate array. Anyone else
    (write-ins, minor filers) is dropped, not summed -- matches the baseline's
    two-candidate convention. Returns (flanagan, craig, other, matched_names)."""
    flanagan = craig = other = 0
    matched = {"flanagan": None, "craig": None}

    for entry in candidate_list or []:
        name = entry.get("name", "")
        votes = int(entry.get("votes") or 0)
        if _match_candidate(name, FLANAGAN_KEYS):
            flanagan += votes
            matched["flanagan"] = name
        elif _match_candidate(name, CRAIG_KEYS):
            craig += votes
            matched["craig"] = name
        else:
            other += votes

    return flanagan, craig, other, matched


def parse_payload(payload: dict, county_names) -> dict:
    """Turn a civicAPI race payload into county-level two-candidate vote counts."""
    lookup = build_county_lookup(county_names)

    state_flanagan, state_craig, state_other, matched_names = extract_two_candidate(
        payload.get("candidates"))

    records = {}
    unmatched = []

    for _slug, region in (payload.get("region_results") or {}).items():
        if str(region.get("type", "")).lower() not in ("county", ""):
            continue
        raw_name = region.get("name", _slug)
        key = normalize_county(raw_name)
        county = lookup.get(key)
        if county is None:
            unmatched.append(raw_name)
            continue

        flanagan, craig, other, _ = extract_two_candidate(region.get("candidates"))
        if flanagan + craig <= 0:
            continue

        records[county] = {
            "flanagan": flanagan,
            "craig": craig,
            "other": other,
            "percent_precincts": region.get("percent_reporting"),
        }

    return {
        "election_name": payload.get("election_name"),
        "last_updated": payload.get("last_updated"),
        "percent_precincts_statewide": payload.get("percent_reporting"),
        "state_flanagan": state_flanagan,
        "state_craig": state_craig,
        "state_other": state_other,
        "candidate_names": matched_names,
        "counties": records,
        "unmatched": unmatched,
    }
