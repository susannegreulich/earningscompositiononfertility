"""
STAGE 3 - Estimate the paper's logit model (Model 2) on the event-occurrence
table and compare the results with the paper. Uses ONLY the stage 2
event-occurrence table, plus the stage 1 parameters as the benchmark.

Two results are produced:
  - the odds ratios, next to the Danish odds ratios in the paper's Table A2;
  - the average marginal probability of conception for each employment-wage
    constellation, as a table and as a figure in the style of the paper's
    Figure 2.

The model is the equation the births were generated from in stage 1, so its
odds ratios should reproduce Table A2 whatever the simulated population looks
like.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix

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
# Stage 0 computes them (main row x Denmark row) and lists them in
# 0dst_data/paper_table_a2_model2_denmark.csv.
PAPER_BASELINE_ODDS = stage1_simulate_registers.FIRST_BIRTH_BASELINE_ODDS
PAPER_ODDS_RATIOS = stage1_simulate_registers.FIRST_BIRTH_ODDS_RATIOS

# The paper's Figure 2, Denmark, Model 2 bars. The paper prints no numbers for
# them, so these are read off the figure by eye and are approximate.
PAPER_FIGURE_2_DENMARK_MODEL_2 = {
    "both_jobless": 0.098, "male_works": 0.113, "female_works": 0.106,
    "dual_male_main_earner": 0.102, "dual_female_main_earner": 0.126, "dual_equal_earners": 0.125}
# Constellations in the order of the paper's Figure 2.
FIGURE_ORDER = ["both_jobless", "male_works", "female_works", "dual_male_main_earner",
                "dual_female_main_earner", "dual_equal_earners"]


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
    fitted = sm.GLM(births_and_non_births, design_matrix, family=sm.families.Binomial()).fit()
    return fitted, design_matrix.design_info


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
            danish_value = PAPER_BASELINE_ODDS
        else:
            variable, _, category = term.rstrip("]").partition("[T.")
            category = category or 1
            danish_value = PAPER_ODDS_RATIOS[variable][category]
        rows.append({"variable": variable, "category": category,
                     "estimated_odds_ratio": np.exp(fitted.params[term]),
                     "lower_95": interval.loc[term, 0], "upper_95": interval.loc[term, 1],
                     "table_a2_danish_odds_ratio": danish_value})
    odds_ratios = pd.DataFrame(rows)
    odds_ratios["paper_within_interval"] = odds_ratios.table_a2_danish_odds_ratio.between(
        odds_ratios.lower_95, odds_ratios.upper_95)
    return odds_ratios


def average_marginal_probabilities(cells, fitted, design_info):
    """The average marginal probability of conception for each constellation.

    What it is: take every couple-year in the sample, pretend the couple had
    this constellation while keeping its other traits (age group, wage tercile,
    education, origin) as they are, predict its probability of conceiving from
    the model, and average over the whole sample. Doing this for each
    constellation in turn gives six probabilities that differ ONLY because of
    the constellation, since the mix of other traits is the same in all six.
    This is what the bars in the paper's Figure 2 show.

    Unlike an odds ratio, the result depends on the sample: a sample with more
    young or low-wage couples has lower probabilities for every constellation.
    """
    cell_weight = cells.couple_years.to_numpy() / cells.couple_years.sum()
    coefficients = fitted.params.to_numpy()
    coefficient_covariance = fitted.cov_params().to_numpy()
    rows = []
    for constellation_name in FIGURE_ORDER:
        # Estimated: every cell gets this constellation, then predict and average.
        as_if = cells.copy()
        as_if["earner_constellation"] = pd.Categorical([constellation_name] * len(cells),
                                                       categories=CONSTELLATIONS)
        design_matrix = np.asarray(build_design_matrices([design_info], as_if)[0])
        probability = 1 / (1 + np.exp(-design_matrix @ coefficients))
        average_probability = cell_weight @ probability
        # Standard error by the delta method: gradient' x covariance x gradient,
        # where the gradient is how the average moves with each coefficient.
        gradient = (cell_weight * probability * (1 - probability)) @ design_matrix
        standard_error = np.sqrt(gradient @ coefficient_covariance @ gradient)
        # Implied by Table A2: the same average, using the odds ratios that
        # stage 1 generated the births from instead of the estimated ones.
        implied_by_table_a2 = cell_weight @ stage1_simulate_registers.birth_probability(
            woman_age_group=cells.woman_age_group.astype(str),
            earner_constellation=[constellation_name] * len(cells),
            household_wage_tercile=cells.household_wage_tercile.astype(str),
            woman_education=cells.woman_education.astype(str),
            either_partner_foreign=cells.either_partner_foreign)
        rows.append({"earner_constellation": constellation_name,
                     "average_marginal_probability": average_probability,
                     "lower_95": average_probability - 1.96 * standard_error,
                     "upper_95": average_probability + 1.96 * standard_error,
                     "implied_by_table_a2_on_this_sample": implied_by_table_a2,
                     "paper_figure_2_approximate": PAPER_FIGURE_2_DENMARK_MODEL_2[constellation_name]})
    return pd.DataFrame(rows)


def plot_average_marginal_probabilities(probabilities, figure_file):
    """Bars with 95% confidence intervals, in the layout of the paper's Figure 2
    (Denmark, Model 2). Diamonds: what Table A2's odds ratios imply on this
    sample. Dashes: the paper's own bars, read off its figure."""
    position = np.arange(len(probabilities))
    figure, axis = plt.subplots(figsize=(9, 4.8), dpi=150)
    axis.bar(position, probabilities.average_marginal_probability, 0.5, color="#2a78d6",
             label="Estimated on the simulated data", zorder=2)
    axis.errorbar(position, probabilities.average_marginal_probability,
                  yerr=probabilities.upper_95 - probabilities.average_marginal_probability,
                  fmt="none", ecolor="#1f1f1e", lw=1, capsize=2, zorder=3)
    axis.scatter(position, probabilities.implied_by_table_a2_on_this_sample, marker="D", s=34,
                 facecolor="white", edgecolor="#1f1f1e", lw=1.3, zorder=4,
                 label="Implied by Table A2's odds ratios on this sample")
    axis.scatter(position + 0.36, probabilities.paper_figure_2_approximate, marker="_", s=260,
                 color="#eb6834", lw=2.5, zorder=4,
                 label="Paper's Figure 2, Denmark, Model 2 (read off the figure)")
    # Labels on two lines, so that neighbouring ones do not run into each other.
    axis.set_xticks(position, [CONSTELLATION_LABELS[c].replace(", ", ",\n")
                               for c in probabilities.earner_constellation], fontsize=9)
    axis.set_ylabel("Average marginal probability of a first birth", fontsize=9)
    axis.set_title("Simulated Denmark: average marginal probability of conception leading to a "
                   "first birth,\nby couples' employment-wage constellation (Model 2)",
                   fontsize=10.5, loc="left")
    axis.set_ylim(0, 0.2)
    axis.grid(axis="y", color="#e4e3dd", lw=0.6, zorder=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors="#6b6a63", labelsize=8.5)
    axis.legend(frameon=False, fontsize=8.5, loc="upper left")
    figure.tight_layout()
    figure.savefig(figure_file)
    plt.close(figure)


def analyse():
    cells = load_cells()
    fitted, design_info = fit_model(cells)
    odds_ratios = odds_ratio_table(fitted)
    probabilities = average_marginal_probabilities(cells, fitted, design_info)

    ESTIMATION_FOLDER.mkdir(exist_ok=True)
    odds_ratios.to_csv(ESTIMATION_FOLDER / "odds_ratios.csv", index=False)
    probabilities.to_csv(ESTIMATION_FOLDER / "average_marginal_probabilities.csv", index=False)
    plot_average_marginal_probabilities(
        probabilities, ESTIMATION_FOLDER / "fig_average_marginal_probabilities.png")

    pd.set_option("display.width", 220)
    print(odds_ratios.round(3).to_string(index=False))
    print(f"  the paper's Danish odds ratio is inside the 95% interval for "
          f"{odds_ratios.paper_within_interval.sum()} of {len(odds_ratios)} estimates")
    print(probabilities.round(3).to_string(index=False))
    print(f"  results: {ESTIMATION_FOLDER}/")


if __name__ == "__main__":
    print("Stage 3: model and comparison ...")
    analyse()
