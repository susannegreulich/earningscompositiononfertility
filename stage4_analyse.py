"""
STAGE 4 - Estimate the paper's logit model (Model 2) on the aggregate table
and compare the odds ratios with the Danish estimates in the paper's Table A2.
Uses ONLY the stage 3 table, plus the stage 1 parameters as the benchmark.

The model is the equation the births were generated from in stage 1, so its
estimates should reproduce Table A2 whatever the simulated population looks
like.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix

import stage1_simulate_registers
from common import (AGE_GROUPS, CONSTELLATION_LABELS, CONSTELLATIONS, EDUCATION_LEVELS,
                    ESTIMATION_FOLDER, EVENT_TABLE_FILE, HOUSEHOLD_WAGE_TERCILES)

# The paper's Model 2 (eq. 1 plus controls): the woman's age group, the
# couple's constellation, household-wage tercile, her education and foreign
# origin.
MODEL_FORMULA = ("woman_age_group + earner_constellation + household_wage_tercile"
                 " + woman_education + either_partner_foreign")
# Category order of each variable; the first category is the reference.
CATEGORY_ORDER = {"woman_age_group": AGE_GROUPS, "earner_constellation": CONSTELLATIONS,
                  "household_wage_tercile": HOUSEHOLD_WAGE_TERCILES,
                  "woman_education": EDUCATION_LEVELS}

# Table A2, Model 2, for Denmark: the values stage 1 generated the births from.
PAPER_BASELINE_ODDS = stage1_simulate_registers.FIRST_BIRTH_BASELINE_ODDS
PAPER_ODDS_RATIOS = stage1_simulate_registers.FIRST_BIRTH_ODDS_RATIOS


def load_cells():
    cells = pd.read_csv(EVENT_TABLE_FILE)
    for variable, categories in CATEGORY_ORDER.items():
        cells[variable] = pd.Categorical(cells[variable], categories=categories)
    return cells


def estimate_odds_ratios(cells):
    """Fit the model and return its odds ratios next to the paper's."""
    # Discrete-time logit hazard, fitted as a binomial model on the cells:
    # first births out of couple-years. This gives exactly the same estimates
    # as a logit on the individual couple-years.
    design_matrix = dmatrix(MODEL_FORMULA, cells, return_type="dataframe")
    births_and_non_births = np.column_stack([cells.first_births,
                                             cells.couple_years - cells.first_births])
    fitted = sm.GLM(births_and_non_births, design_matrix, family=sm.families.Binomial()).fit()

    # The coefficients are log odds; exponentiate them and their 95% interval.
    interval = np.exp(fitted.conf_int())
    rows = []
    for term in fitted.params.index:
        # Terms are named "Intercept", "variable[T.category]", or just
        # "variable" for the 0/1 foreign-origin indicator.
        if term == "Intercept":
            variable, category = "baseline odds", ""
            paper_value = PAPER_BASELINE_ODDS
        else:
            variable, _, category = term.rstrip("]").partition("[T.")
            category = category or 1
            paper_value = PAPER_ODDS_RATIOS[variable][category]
        rows.append({"variable": variable, "category": category,
                     "odds_ratio": np.exp(fitted.params[term]),
                     "lower_95": interval.loc[term, 0], "upper_95": interval.loc[term, 1],
                     "odds_ratio_in_paper": paper_value})
    odds_ratios = pd.DataFrame(rows)
    odds_ratios["paper_within_interval"] = odds_ratios.odds_ratio_in_paper.between(
        odds_ratios.lower_95, odds_ratios.upper_95)
    return odds_ratios


def plot_constellation_odds_ratios(odds_ratios, figure_file):
    """Bars for the estimates with 95% confidence intervals; white diamonds for
    the paper's Danish estimates."""
    shown = CONSTELLATIONS[1:]                    # all but the reference
    position = np.arange(len(shown))
    estimates = (odds_ratios[odds_ratios.variable == "earner_constellation"]
                 .set_index("category").loc[shown])
    figure, axis = plt.subplots(figsize=(9, 4.8), dpi=150)
    axis.bar(position, estimates.odds_ratio, 0.5, color="#2a78d6", label="Estimated", zorder=2)
    axis.errorbar(position, estimates.odds_ratio,
                  yerr=[estimates.odds_ratio - estimates.lower_95,
                        estimates.upper_95 - estimates.odds_ratio],
                  fmt="none", ecolor="#1f1f1e", lw=1, capsize=2, zorder=3)
    axis.scatter(position, estimates.odds_ratio_in_paper, marker="D", s=34, facecolor="white",
                 edgecolor="#1f1f1e", lw=1.3, zorder=4, label="Paper, Table A2 (Denmark)")
    axis.axhline(1, color="#6b6a63", lw=0.8)
    axis.set_xticks(position, [CONSTELLATION_LABELS[c] for c in shown], fontsize=9)
    axis.set_ylabel("Odds ratio of conceiving a first child", fontsize=9)
    axis.set_title("Simulated Denmark: odds ratios vs. dual earners with equal wages",
                   fontsize=11, loc="left")
    axis.grid(axis="y", color="#e4e3dd", lw=0.6, zorder=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors="#6b6a63", labelsize=8.5)
    axis.legend(frameon=False, fontsize=8.5, loc="lower right")
    figure.tight_layout()
    figure.savefig(figure_file)
    plt.close(figure)


def analyse():
    cells = load_cells()
    odds_ratios = estimate_odds_ratios(cells)
    ESTIMATION_FOLDER.mkdir(exist_ok=True)
    odds_ratios.to_csv(ESTIMATION_FOLDER / "odds_ratios.csv", index=False)
    plot_constellation_odds_ratios(odds_ratios, ESTIMATION_FOLDER / "fig_odds_ratios.png")

    pd.set_option("display.width", 200)
    print(odds_ratios.round(3).to_string(index=False))
    print(f"  the paper's value is inside the 95% interval for "
          f"{odds_ratios.paper_within_interval.sum()} of {len(odds_ratios)} estimates")
    print(f"  results: {ESTIMATION_FOLDER}/")


if __name__ == "__main__":
    print("Stage 4: model and comparison ...")
    analyse()
