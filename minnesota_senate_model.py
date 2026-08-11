"""
Minnesota US Senate Democratic Primary -- live county-level model.
Angie Craig vs. Peggy Flanagan, two-candidate (per Wilson's instruction).

NOT DEDUCTIVE. This is the one deliberate architectural break from the
Michigan/Wisconsin/South Dakota template family, per Wilson's explicit
instruction: "not as deductive... blend projections with current results
unless there is a consistent pattern that projections are off and then use
current results in that case."

WHAT "DEDUCTIVE" MEANS AND WHY THIS ISN'T THAT

The WI/SD/GA template holds a county's counted votes as literal, unconditional
truth and only models the UNCOUNTED remainder:
    projected_votes = counted_votes + remaining_votes * blended_rate
That's correct when you trust every partial batch that comes in as an
unbiased (if incomplete) sample -- Wisconsin's counties report in no
consistent order, so there's no reason to distrust an early batch's rate,
only its completeness.

Wilson's instruction here is different: trust the BASELINE too, not just
completeness-weighted counted votes. So instead of "counted + blended
remainder", every county's full projected result is:
    projected_votes = effective_turnout * blended_share
where blended_share comes from a credibility-weighted average of the
county's OWN observed margin and its (shift-adjusted) baseline margin --
credibility grows with how much of the county has reported, same shape as
the WI template's `credibility` property, but it's now applied to the WHOLE
county's projection, not just the leftover votes. A raw early batch that
disagrees hard with the baseline pulls the projection only partway, not all
the way -- proportional to how much of the county is actually in.

THE "CONSISTENT PATTERN" ESCAPE HATCH

The statewide/regional shift machinery (`_recompute_shifts`, unchanged in
spirit from the WI template) is what "unless there's a consistent pattern
that projections are off" cashes out to. It's evidence-weighted empirical-
Bayes shrinkage: one wild county barely moves the shift (GLOBAL_EVIDENCE_PRIOR
dominates when total weight is low), but if MANY counties -- weighted by
credibility, down-weighted if they're outliers relative to each other -- show
the same directional surprise, the shift converges toward that surprise's
full size, and gets folded into every county's baseline before the
observed/baseline blend even happens. So a confirmed statewide pattern
doesn't just nudge the model, it silently re-centers the "baseline" every
county's own results get blended against, county-by-county evidence on top of
that. A MOMENTUM constraint additionally hard-clamps a well-reported county's
projection to within MOMENTUM_MAX_DRIFT points of its own observed margin
once it clears MOMENTUM_TRIGGER_PCT reporting, so a strong true county-level
signal can never be blended away entirely by a stale baseline.

WHY BASELINE TRUST IS THE RIGHT DEFAULT HERE

Wilson's basis for this call: the MN baseline (minnesota_governor_model.py)
isn't a rough guess -- it's built from an actual crowd-prediction map read
pixel-by-pixel into a 10-step confidence tier per county, plus a 2020
Sanders/Warren coalition-shape covariate, both real signals rather than
placeholders. That's worth defending against a noisy early batch in a way a
placeholder baseline wouldn't be.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

CANDIDATES = ("flanagan", "craig")

BASELINE_PATH = "mn_county_baseline.csv"
TARGET_TURNOUT = 500_000

# ------------------------------------------------------------------
# Regions -- media-market-ish groupings of MN's 87 counties, used only for
# the regional layer of shift shrinkage (see UNIVERSAL_TEMPLATE_GUIDE.md).
# ------------------------------------------------------------------
REGIONS = {
    "Metro": ["Anoka", "Carver", "Dakota", "Hennepin", "Ramsey", "Scott", "Washington"],
    "Southeast": ["Dodge", "Fillmore", "Freeborn", "Goodhue", "Houston", "Mower",
                  "Olmsted", "Rice", "Steele", "Wabasha", "Winona"],
    "Southwest": ["Cottonwood", "Faribault", "Jackson", "Lac qui Parle", "Lincoln",
                  "Lyon", "Martin", "Murray", "Nobles", "Pipestone", "Redwood",
                  "Renville", "Rock", "Watonwan", "Yellow Medicine"],
    "South Central": ["Blue Earth", "Brown", "Le Sueur", "McLeod", "Nicollet",
                       "Sibley", "Waseca"],
    "Central": ["Benton", "Big Stone", "Chippewa", "Douglas", "Grant", "Kandiyohi",
                "Meeker", "Pope", "Sherburne", "Stearns", "Stevens", "Swift",
                "Todd", "Traverse", "Wright"],
    "Northwest": ["Becker", "Beltrami", "Clay", "Clearwater", "Hubbard", "Kittson",
                  "Lake of the Woods", "Mahnomen", "Marshall", "Norman",
                  "Otter Tail", "Pennington", "Polk", "Red Lake", "Roseau", "Wilkin"],
    "North Central": ["Aitkin", "Cass", "Chisago", "Crow Wing", "Isanti", "Kanabec",
                       "Mille Lacs", "Morrison", "Pine", "Wadena"],
    "Arrowhead": ["Carlton", "Cook", "Itasca", "Koochiching", "Lake", "St. Louis"],
}
COUNTY_REGION = {c: r for r, cs in REGIONS.items() for c in cs}

# Counties big/internally-diverse enough that a partial count is a biased
# draw, not a random sample -- same treatment as WI's Milwaukee/Dane.
COUNTY_HETEROGENEITY = {
    "DEFAULT": 2.5,
    "Hennepin": 12.0,
    "Ramsey": 8.0,
    "St. Louis": 5.0,
}

# ------------------------------------------------------------------
# Tuning constants -- same values as the WI template except where the
# not-deductive rework required a genuinely different meaning (flagged below).
# ------------------------------------------------------------------
CREDIBILITY_EXPONENT = 2.0
OUTLIER_LAMBDA = 3.0
TAU_FLOOR = 0.08
N_SIMS = 20000

TURNOUT_FULL_TRUST_PCT = 25.0
TURNOUT_CLAMP = (0.40, 2.50)

MOMENTUM_TRIGGER_PCT = 0.30
MOMENTUM_MAX_DRIFT = 10.0

MAX_SINGLE_COUNTY_SHARE = 0.25    # no single county's own weight can count for
                                  # more than this share of GLOBAL_EVIDENCE_PRIOR
                                  # -- otherwise one huge county alone (Hennepin
                                  # reporting early) reads as a "consistent
                                  # pattern" on its own, which it isn't
GLOBAL_EVIDENCE_PRIOR = 60_000.0   # enough to resist one big county's early
                                   # partial count, but lets a true broad-based
                                   # pattern (most of the state reporting
                                   # consistently) dominate well before 100%
REGIONAL_EVIDENCE_PRIOR = 8_000.0

PRE_ELECTION_MARGIN_SD = 9.0      # retune against a target pre-election win prob


def _load_baseline(path: str = BASELINE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(df["county"]) - set(COUNTY_REGION)
    if missing:
        raise ValueError(f"No region for: {missing}")
    return df


@dataclass
class CountyState:
    name: str
    region: str
    baseline_margin: float          # Flanagan minus Craig, points, from the baseline CSV
    expected_turnout: int           # original baseline turnout, never mutated
    calibrated_turnout: Optional[float] = None
    pct_reporting: float = 0.0      # PRECINCT reporting -- diagnostic only, not completeness
    votes: Dict[str, int] = field(default_factory=lambda: {"flanagan": 0, "craig": 0})

    @property
    def effective_turnout(self) -> float:
        return self.calibrated_turnout if self.calibrated_turnout is not None else self.expected_turnout

    @property
    def counted_votes(self) -> int:
        return self.votes["flanagan"] + self.votes["craig"]

    @property
    def pct_counted(self) -> float:
        """Fraction of EFFECTIVE TURNOUT counted so far -- the real completeness
        signal used for credibility, as opposed to pct_reporting (precincts)."""
        if self.effective_turnout <= 0:
            return 0.0
        return min(1.0, self.counted_votes / self.effective_turnout)

    @property
    def observed_margin(self) -> Optional[float]:
        cv = self.counted_votes
        if cv <= 0:
            return None
        return 100.0 * (self.votes["flanagan"] - self.votes["craig"]) / cv

    @property
    def heterogeneity(self) -> float:
        return COUNTY_HETEROGENEITY.get(self.name, COUNTY_HETEROGENEITY["DEFAULT"])

    @property
    def credibility(self) -> float:
        """How much weight the county's OWN observed margin gets against the
        (shift-adjusted) baseline -- grows with how much of the county's
        effective turnout has actually been counted, same curve shape as the
        WI template, penalized for big/heterogeneous counties early on."""
        p = self.pct_counted
        if p <= 0:
            return 0.0
        completeness_weight = p ** (1 / CREDIBILITY_EXPONENT)
        design_var = (self.heterogeneity ** 2) * (1 - p)
        noise_penalty = 1.0 / (1.0 + design_var / 50.0)
        return min(0.995, completeness_weight * noise_penalty)


class MinnesotaSenateModel:
    def __init__(self, baseline_path: str = BASELINE_PATH):
        df = _load_baseline(baseline_path)
        self.counties: Dict[str, CountyState] = {}
        for _, row in df.iterrows():
            self.counties[row["county"]] = CountyState(
                name=row["county"],
                region=COUNTY_REGION[row["county"]],
                baseline_margin=float(row["margin"]),
                expected_turnout=int(row["turnout"]),
            )
        self.total_evidence_weight = 0.0
        self.statewide_shift = 0.0
        self.statewide_shift_var = TAU_FLOOR ** 2
        self.regional_shift: Dict[str, float] = {r: 0.0 for r in REGIONS}
        self.turnout_pooled_ratio = 1.0

    # ------------------------------------------------------------
    def update_county(self, name: str, flanagan_votes: int, craig_votes: int,
                      pct_reporting: Optional[float]) -> None:
        c = self.counties[name]
        c.votes = {"flanagan": int(flanagan_votes or 0), "craig": int(craig_votes or 0)}
        c.pct_reporting = (pct_reporting or 0.0) / 100.0 if pct_reporting and pct_reporting > 1 else (pct_reporting or 0.0)

    # ------------------------------------------------------------
    def _recalibrate_turnout(self) -> None:
        """Feed-implied turnout from percent_reporting (PRECINCTS, not votes --
        same caution as the MI/WI models), credibility-ramped and clamped, then
        the pooled ratio propagates to counties still at zero. See
        turnout_calibration.py's docstring in the MI build for the full
        reasoning; same mechanism here, condensed."""
        rows = []
        for c in self.counties.values():
            if c.pct_reporting <= 0 or c.counted_votes <= 0:
                c.calibrated_turnout = None
                continue
            implied = c.counted_votes / c.pct_reporting
            raw_ratio = implied / c.expected_turnout
            clamped_ratio = float(np.clip(raw_ratio, *TURNOUT_CLAMP))
            weight = float(np.clip(c.pct_reporting * 100 / TURNOUT_FULL_TRUST_PCT, 0.0, 1.0))
            final_ratio = (1 - weight) * 1.0 + weight * clamped_ratio
            c.calibrated_turnout = c.expected_turnout * final_ratio
            rows.append((c.name, raw_ratio, c.expected_turnout))

        if len(rows) >= 5:
            ratios = np.clip([r[1] for r in rows], *TURNOUT_CLAMP)
            sizes = np.array([r[2] for r in rows], dtype=float)
            order = np.argsort(ratios)
            cumulative = np.cumsum(sizes[order]) / sizes.sum()
            median_idx = order[int(np.searchsorted(cumulative, 0.5))]
            self.turnout_pooled_ratio = float(ratios[median_idx])

            reporting_names = {r[0] for r in rows}
            reporting_turnout = sum(self.counties[n].expected_turnout for n in reporting_names)
            total_turnout = sum(c.expected_turnout for c in self.counties.values())
            strength = float(np.clip(reporting_turnout / total_turnout, 0.0, 1.0))
            applied = 1.0 + (self.turnout_pooled_ratio - 1.0) * strength
            for c in self.counties.values():
                if c.calibrated_turnout is None:
                    c.calibrated_turnout = c.expected_turnout * applied

    # ------------------------------------------------------------
    def _recompute_shifts(self) -> None:
        """Empirical-Bayes shrinkage of (observed margin - baseline margin)
        across reporting counties -- this is the 'consistent pattern' engine.
        Outlier counties (relative to the reporting group) are down-weighted
        so one wild batch can't pass for a statewide pattern; if many counties
        genuinely agree, the shrinkage denominator stops being dominated by
        the prior and the shift converges toward the real size of the swing."""
        names, surprises, weights, regions = [], [], [], []
        for c in self.counties.values():
            om = c.observed_margin
            if om is None:
                continue
            w = c.counted_votes * (c.pct_counted ** (1 / CREDIBILITY_EXPONENT))
            names.append(c.name)
            surprises.append(om - c.baseline_margin)
            weights.append(w)
            regions.append(c.region)

        if not surprises:
            self.statewide_shift = 0.0
            self.regional_shift = {r: 0.0 for r in REGIONS}
            self.total_evidence_weight = 0.0
            return

        surprises = np.array(surprises)
        weights = np.array(weights, dtype=float)
        regions = np.array(regions)

        base_w = weights
        if len(surprises) > 1:
            wmean0 = np.average(surprises, weights=base_w) if base_w.sum() > 0 else surprises.mean()
            resid = surprises - wmean0
            scale = max(np.std(resid), 1e-6)
            outlier_factor = 1.0 / (1.0 + (np.abs(resid) / (OUTLIER_LAMBDA * scale)) ** 2)
        else:
            outlier_factor = np.ones_like(surprises)

        w = base_w * outlier_factor
        # No single county's evidence weight may exceed this share of the
        # global prior -- see MAX_SINGLE_COUNTY_SHARE above. This is what
        # actually separates "one big county swung early" from "a real
        # multi-county pattern": both can carry the same total weight, but
        # only the multi-county case survives this cap undiminished.
        w = np.minimum(w, MAX_SINGLE_COUNTY_SHARE * GLOBAL_EVIDENCE_PRIOR)
        self.total_evidence_weight = float(w.sum())

        if w.sum() <= 0:
            self.statewide_shift = 0.0
        else:
            wmean = np.average(surprises, weights=w)
            tau2 = max(TAU_FLOOR ** 2, np.average((surprises - wmean) ** 2, weights=w))
            shrink = w.sum() / (w.sum() + GLOBAL_EVIDENCE_PRIOR)
            self.statewide_shift = shrink * wmean
            self.statewide_shift_var = tau2

        for region in REGIONS:
            idx = regions == region
            if not idx.any() or w[idx].sum() <= 0:
                self.regional_shift[region] = self.statewide_shift
                continue
            r_wmean = np.average(surprises[idx], weights=w[idx])
            shrink = w[idx].sum() / (w[idx].sum() + REGIONAL_EVIDENCE_PRIOR)
            self.regional_shift[region] = shrink * r_wmean + (1 - shrink) * self.statewide_shift

    # ------------------------------------------------------------
    def project_margin(self, c: CountyState) -> float:
        """The core not-deductive step. adjusted_baseline folds in the
        confirmed-pattern shift; the county's OWN observed margin is then
        blended in at `credibility` weight -- not held fixed, not ignored,
        blended. Once well-reported, MOMENTUM clamps the blend within
        MOMENTUM_MAX_DRIFT of what's actually been counted, so a confirmed
        county-level result can't be diluted away by a stale baseline."""
        adjusted_baseline = c.baseline_margin + self.statewide_shift + \
            (self.regional_shift[c.region] - self.statewide_shift)
        adjusted_baseline = float(np.clip(adjusted_baseline, -60.0, 60.0))

        om = c.observed_margin
        if om is None:
            return adjusted_baseline

        w = c.credibility
        blended = w * om + (1 - w) * adjusted_baseline

        if c.pct_counted >= MOMENTUM_TRIGGER_PCT:
            blended = float(np.clip(blended, om - MOMENTUM_MAX_DRIFT, om + MOMENTUM_MAX_DRIFT))

        return float(np.clip(blended, -60.0, 60.0))

    # ------------------------------------------------------------
    def project(self) -> Dict:
        """Full projection cycle: recalibrate turnout, recompute shifts, then
        allocate EVERY county's full effective turnout via its blended
        margin (not counted-plus-remainder -- see module docstring)."""
        self._recalibrate_turnout()
        self._recompute_shifts()

        flanagan_total = craig_total = 0.0
        counted_flanagan = counted_craig = 0
        n_reported = 0
        for c in self.counties.values():
            margin = self.project_margin(c)
            turnout = c.effective_turnout
            flanagan_total += turnout * (50 + margin / 2) / 100
            craig_total += turnout * (50 - margin / 2) / 100
            counted_flanagan += c.votes["flanagan"]
            counted_craig += c.votes["craig"]
            if c.counted_votes > 0:
                n_reported += 1

        grand_total = flanagan_total + craig_total
        projected_turnout = sum(c.effective_turnout for c in self.counties.values())
        pct_counted = (counted_flanagan + counted_craig) / projected_turnout if projected_turnout else 0.0

        return {
            "flanagan_pct": 100 * flanagan_total / grand_total,
            "craig_pct": 100 * craig_total / grand_total,
            "flanagan_votes": flanagan_total,
            "craig_votes": craig_total,
            "counted_flanagan": counted_flanagan,
            "counted_craig": counted_craig,
            "n_reported": n_reported,
            "projected_turnout": projected_turnout,
            "pct_counted": pct_counted,
            "statewide_shift": self.statewide_shift,
            "regional_shift": dict(self.regional_shift),
            "turnout_pooled_ratio": self.turnout_pooled_ratio,
            "total_evidence_weight": self.total_evidence_weight,
        }

    # ------------------------------------------------------------
    def run_simulation(self, n_sims: int = N_SIMS, seed: Optional[int] = None) -> Dict:
        """Vectorized Monte Carlo around the blended point projection. Per-
        county noise shrinks with credibility (a well-blended county is
        already anchored near its true result); a shared statewide shock
        reflects how uncertain the shift estimate itself still is."""
        self._recalibrate_turnout()
        self._recompute_shifts()

        rng = np.random.default_rng(seed)
        counties = list(self.counties.values())
        n = len(counties)

        cred = np.array([c.credibility for c in counties])
        heterog = np.array([c.heterogeneity for c in counties])
        eff_turnout = np.array([c.effective_turnout for c in counties])
        point_margin = np.array([self.project_margin(c) for c in counties])
        pct_counted = np.array([c.pct_counted for c in counties])

        base_sd = 8.0
        county_sd = base_sd * (1 - cred) ** 0.5 + heterog * (1 - cred) * 0.3
        county_sd = np.maximum(county_sd, 0.5)

        evidence_shrink = self.total_evidence_weight / (self.total_evidence_weight + GLOBAL_EVIDENCE_PRIOR)
        prior_sd = PRE_ELECTION_MARGIN_SD * (1 - evidence_shrink)
        statewide_sd = math.sqrt(self.statewide_shift_var) * 15.0
        statewide_sd = math.sqrt(statewide_sd ** 2 + prior_sd ** 2)

        obs_arr = np.array([(c.observed_margin if c.observed_margin is not None else 0.0)
                            for c in counties])
        momentum_active = pct_counted >= MOMENTUM_TRIGGER_PCT
        lo_bound = obs_arr - MOMENTUM_MAX_DRIFT
        hi_bound = obs_arr + MOMENTUM_MAX_DRIFT

        shared_shock = rng.normal(0, statewide_sd, size=(n_sims, 1))
        county_shock = rng.normal(0, 1, size=(n_sims, n)) * county_sd[None, :]
        sim_margin = point_margin[None, :] + shared_shock + county_shock

        clipped = np.clip(sim_margin, lo_bound[None, :], hi_bound[None, :])
        sim_margin = np.where(momentum_active[None, :], clipped, sim_margin)
        sim_margin = np.clip(sim_margin, -60.0, 60.0)

        flanagan_votes = eff_turnout[None, :] * (50 + sim_margin / 2) / 100
        craig_votes = eff_turnout[None, :] - flanagan_votes
        totals_fl = flanagan_votes.sum(axis=1)
        totals_cr = craig_votes.sum(axis=1)
        grand = totals_fl + totals_cr
        margins = 100 * totals_fl / grand - 100 * totals_cr / grand

        return {
            "n_sims": n_sims,
            "mean_margin": float(np.mean(margins)),
            "median_margin": float(np.median(margins)),
            "p05": float(np.percentile(margins, 5)), "p10": float(np.percentile(margins, 10)),
            "p25": float(np.percentile(margins, 25)), "p75": float(np.percentile(margins, 75)),
            "p90": float(np.percentile(margins, 90)), "p95": float(np.percentile(margins, 95)),
            "flanagan_win_prob": float(np.mean(margins > 0)),
            "craig_win_prob": float(np.mean(margins < 0)),
        }


if __name__ == "__main__":
    model = MinnesotaSenateModel()
    proj = model.project()
    print("PRE-ELECTION (no votes counted)")
    print(f"  Flanagan {proj['flanagan_pct']:.2f}  Craig {proj['craig_pct']:.2f}"
          f"  (turnout {proj['projected_turnout']:,.0f})")
    sim = model.run_simulation(seed=42)
    print(f"  Mean margin {sim['mean_margin']:+.2f}  "
          f"90% CI [{sim['p05']:+.1f}, {sim['p95']:+.1f}]  "
          f"Flanagan win prob {sim['flanagan_win_prob']:.1%}")

    print()
    print("SIMULATED ELECTION NIGHT: Hennepin reports 40% in, "
          "running 8 points more Flanagan than baseline")
    hennepin = model.counties["Hennepin"]
    hen_margin = hennepin.baseline_margin + 8.0
    hen_turnout = hennepin.expected_turnout
    hen_counted = int(hen_turnout * 0.40)
    hen_fl = int(hen_counted * (50 + hen_margin / 2) / 100)
    hen_cr = hen_counted - hen_fl
    model.update_county("Hennepin", hen_fl, hen_cr, pct_reporting=35.0)
    proj = model.project()
    print(f"  Hennepin credibility: {hennepin.credibility:.3f}  "
          f"observed margin: {hennepin.observed_margin:+.2f}  "
          f"blended margin: {model.project_margin(hennepin):+.2f}")
    print(f"  Statewide: Flanagan {proj['flanagan_pct']:.2f}  Craig {proj['craig_pct']:.2f}")
    print(f"  Statewide shift: {proj['statewide_shift']:+.3f}  "
          f"(one county alone shouldn't move this much: evidence prior = {GLOBAL_EVIDENCE_PRIOR})")
