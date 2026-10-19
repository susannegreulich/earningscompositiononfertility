"""
STAGE 3 - Estimate the paper's logit model (Model 2) on the event-occurrence
table and compare the odds ratios with the Danish odds ratios of the paper's
Table A2, as a table and as a chart. A second chart shows the odds themselves
for each constellation, written as fractions ("1 to 13"). Uses ONLY the stage 2
event-occurrence table, plus the Danish Table A2 values in 0data/, which stage 1
generated the births from.

The model is the equation the births were generated from, so its odds ratios
should reproduce Table A2 whatever the simulated population looks like.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix

from common import (CONSTELLATION_LABELS, CONSTELLATIONS, DANISH_BASELINE_ODDS, DANISH_ODDS_RATIOS,
                    ESTIMATION_FOLDER, EVENT_TABLE_FILE)

# The paper's Model 2 (eq. 1 plus controls): the woman's age group, the
# couple's constellation, household-wage tercile, her education and foreign
# origin.
MODEL_FORMULA = ("woman_age_group + earner_constellation + household_wage_tercile"
                 " + woman_education + either_partner_foreign")
# Category order of each variable, as in the table of the paper's odds
# ratios; the first category is the reference. Foreign origin is a 0/1 number
# and needs no order.
CATEGORY_ORDER = {variable: list(odds_ratios) for variable, odds_ratios in DANISH_ODDS_RATIOS.items()
                  if variable != "either_partner_foreign"}


def load_cells():
    cells = pd.read_csv(EVENT_TABLE_FILE)
    for variable, categories in CATEGORY_ORDER.items():
        cells[variable] = pd.Categorical(cells[variable], categories=categories)
    return cells


def fit_model(cells):
    """Discrete-time logit hazard, fitted as a binomial model on the cells:
    first births out of couple-years. This gives exactly the same estimates as
    a logit on the individual couple-years."""
    design_matrix = dmatrix(MODEL_FORMULA, cells, return_type="dataframe")
    births_and_non_births = np.column_stack([cells.first_births,
                                             cells.couple_years - cells.first_births])
    return sm.GLM(births_and_non_births, design_matrix, family=sm.families.Binomial()).fit()


def odds_ratio_table(fitted):
    """The estimated odds ratios next to the paper's Table A2."""
    # The coefficients are log odds; exponentiate them and their 95% interval.
    interval = np.exp(fitted.conf_int())
    rows = []
    for term in fitted.params.index:
        # Terms are named "Intercept", "variable[T.category]", or just
        # "variable" for the 0/1 foreign-origin indicator.
        if term == "Intercept":
            variable, category = "baseline odds", ""
            danish_value = DANISH_BASELINE_ODDS
        else:
            variable, _, category = term.rstrip("]").partition("[T.")
            category = category or 1
            danish_value = DANISH_ODDS_RATIOS[variable][category]
        rows.append({"variable": variable, "category": category,
                     "estimated_odds_ratio": np.exp(fitted.params[term]),
                     "lower_95": interval.loc[term, 0], "upper_95": interval.loc[term, 1],
                     "table_a2_danish_odds_ratio": danish_value})
    odds_ratios = pd.DataFrame(rows)
    odds_ratios["paper_within_interval"] = odds_ratios.table_a2_danish_odds_ratio.between(
        odds_ratios.lower_95, odds_ratios.upper_95)
    return odds_ratios


def odds_by_constellation(fitted):
    """The ODDS of conceiving (not the odds ratio) for each constellation.

    Odds = couples that conceive for each couple that does not. They are given
    for one fixed couple, the reference couple (woman 18-25, low wage tercile,
    low education, neither partner born abroad), in each constellation:

        odds = baseline odds x the constellation's odds ratio

    For dual equal earners, the reference constellation, the odds ratio is 1,
    so the odds are the baseline odds themselves. Odds of 0.077 are the same as
    "1 to 13": 1 / 0.077 = 13 couples that do not conceive for each one that does.
    """
    covariance = fitted.cov_params()
    rows = []
    # The reference constellation last, as in the paper's figures.
    for constellation_name in CONSTELLATIONS[1:] + CONSTELLATIONS[:1]:
        term = f"earner_constellation[T.{constellation_name}]"
        # On the log scale the odds are a sum: intercept + the constellation's
        # coefficient (nothing is added for the reference constellation).
        log_odds = fitted.params["Intercept"]
        variance = covariance.loc["Intercept", "Intercept"]
        if term in fitted.params:
            log_odds += fitted.params[term]
            # Variance of a sum of two estimates: var(a) + var(b) + 2 cov(a, b).
            variance += covariance.loc[term, term] + 2 * covariance.loc["Intercept", term]
        odds = np.exp(log_odds)
        paper_odds = DANISH_BASELINE_ODDS * DANISH_ODDS_RATIOS["earner_constellation"][constellation_name]
        rows.append({"earner_constellation": constellation_name,
                     "estimated_odds": odds,
                     "lower_95": np.exp(log_odds - 1.96 * np.sqrt(variance)),
                     "upper_95": np.exp(log_odds + 1.96 * np.sqrt(variance)),
                     "estimated_odds_as_fraction": f"1 to {1 / odds:.1f}",
                     "table_a2_danish_odds": paper_odds,
                     "table_a2_danish_odds_as_fraction": f"1 to {1 / paper_odds:.1f}"})
    return pd.DataFrame(rows)


def bar_chart(figure_file, title, y_label, labels, estimate, lower, upper, paper,
              line_at_one=False, text_above_bars=None):
    """The chart used for every result: one bar per group for the estimate, its
    95% confidence interval, and a white diamond for the paper's Danish value."""
    position = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(max(5.5, 1.7 * len(labels)), 4.9), dpi=150)
    axis.bar(position, estimate, 0.5, color="#2a78d6", label="Estimated on the simulated data", zorder=2)
    axis.errorbar(position, estimate, yerr=[estimate - lower, upper - estimate],
                  fmt="none", ecolor="#1f1f1e", lw=1, capsize=2, zorder=3)
    axis.scatter(position, paper, marker="D", s=34, facecolor="white", edgecolor="#1f1f1e",
                 lw=1.3, zorder=4, label="Paper, Table A2, Denmark")
    if line_at_one:
        # An odds ratio of 1 means the same odds as the reference category.
        axis.axhline(1, color="#6b6a63", lw=0.8)
    if text_above_bars is not None:
        for x, top, text in zip(position, upper, text_above_bars):
            axis.text(x, top * 1.03, text, ha="center", va="bottom", fontsize=10, color="#0b0b0b")
    axis.set_xticks(position, labels, fontsize=9)
    axis.set_ylabel(y_label, fontsize=9)
    axis.set_title(title, fontsize=10, loc="left")
    # Room above the bars for the legend, so that it does not cover them.
    axis.set_ylim(0, max(1.0 if line_at_one else 0, upper.max()) * 1.4)
    axis.grid(axis="y", color="#e4e3dd", lw=0.6, zorder=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors="#6b6a63", labelsize=8.5)
    axis.legend(frameon=False, fontsize=8.5, loc="upper left")
    figure.tight_layout()
    figure.savefig(figure_file)
    plt.close(figure)


# Constellation labels on two lines, so that neighbouring ones do not run into each other.
CONSTELLATION_CHART_LABELS = {name: label.replace(", ", ",\n")
                              for name, label in CONSTELLATION_LABELS.items()}
# The odds-ratio charts: one per variable. For each: the file name, the title,
# what the bars are compared with, and a readable label for each category.
ODDS_RATIO_CHARTS = {
    "earner_constellation": {
        "file_name": "fig_odds_ratios_by_constellation.png",
        "title": "odds ratios by employment-wage constellation",
        "reference": "dual earners with equal wages",
        "labels": CONSTELLATION_CHART_LABELS},
    "household_wage_tercile": {
        "file_name": "fig_odds_ratios_by_household_wages.png",
        "title": "odds ratios by household wages",
        "reference": "couples in the low third of household wages",
        "labels": {"middle": "Middle third", "high": "High third"}},
    "woman_age_group": {
        "file_name": "fig_odds_ratios_by_age.png",
        "title": "odds ratios by the woman's age",
        "reference": "couples where the woman is aged 18-25",
        "labels": {"26-30": "26-30", "31-35": "31-35", "36-45": "36-45"}},
}


def plot_odds_ratios(odds_ratios, variable):
    """One variable's odds ratios. The reference category has no bar: it is the line at 1."""
    chart = ODDS_RATIO_CHARTS[variable]
    estimates = odds_ratios[odds_ratios.variable == variable]
    bar_chart(ESTIMATION_FOLDER / chart["file_name"],
              f"Simulated Denmark: {chart['title']} (Model 2)\nrelative to {chart['reference']}",
              "Odds ratio of conceiving a first child",
              [chart["labels"][str(category)] for category in estimates.category],
              estimates.estimated_odds_ratio, estimates.lower_95, estimates.upper_95,
              estimates.table_a2_danish_odds_ratio, line_at_one=True)


def plot_odds_by_constellation(odds):
    """The odds of each constellation, each bar labelled as a fraction ("1 to 13")."""
    bar_chart(ESTIMATION_FOLDER / "fig_odds_by_constellation.png",
              "Simulated Denmark: odds of conceiving a first child within a year, by constellation "
              "(Model 2)\nfor a couple with the woman aged 18-25, low household wages, low "
              "education, neither born abroad",
              "Odds: couples that conceive per couple that does not",
              [CONSTELLATION_CHART_LABELS[name] for name in odds.earner_constellation],
              odds.estimated_odds, odds.lower_95, odds.upper_95, odds.table_a2_danish_odds,
              text_above_bars=odds.estimated_odds_as_fraction)


def analyse():
    cells = load_cells()
    fitted = fit_model(cells)
    odds_ratios = odds_ratio_table(fitted)
    odds = odds_by_constellation(fitted)

    ESTIMATION_FOLDER.mkdir(exist_ok=True)
    odds_ratios.to_csv(ESTIMATION_FOLDER / "odds_ratios.csv", index=False)
    for variable in ODDS_RATIO_CHARTS:
        plot_odds_ratios(odds_ratios, variable)
    odds.to_csv(ESTIMATION_FOLDER / "odds_by_constellation.csv", index=False)
    plot_odds_by_constellation(odds)

    pd.set_option("display.width", 220)
    print(odds_ratios.round(3).to_string(index=False))
    print(f"  the paper's Danish odds ratio is inside the 95% interval for "
          f"{odds_ratios.paper_within_interval.sum()} of {len(odds_ratios)} estimates")
    print(odds.round(4).to_string(index=False))
    print(f"  results: {ESTIMATION_FOLDER}/")


if __name__ == "__main__":
    print("Stage 3: model and comparison ...")
    analyse()
