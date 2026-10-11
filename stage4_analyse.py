"""
STAGE 4 - Estimate the paper's models on the aggregate table and compare the
results with (a) the truth set in stage 1 and (b) the paper's published Danish
numbers. Uses ONLY the stage 3 table, plus the stage 1 parameters for the truth.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix
from scipy.special import expit

import stage1_simulate_registers
from common import (AGE_GROUPS, CONSTELLATION_LABELS, CONSTELLATIONS, EDUCATION_LEVELS,
                    EVENT_TABLE_FILE, HOUSEHOLD_WAGE_TERCILES, OUTPUT_FOLDER)

# The paper's two models. Model 1 controls only for the woman's age group;
# Model 2 adds household-wage tercile, her education and foreign origin.
MODEL_FORMULAS = {
    "model1": "woman_age_group + earner_constellation",
    "model2": ("woman_age_group + earner_constellation + household_wage_tercile"
               " + woman_education + either_partner_foreign")}
# Category order of each variable; the first category is the reference.
CATEGORY_ORDER = {"woman_age_group": AGE_GROUPS, "earner_constellation": CONSTELLATIONS,
                  "household_wage_tercile": HOUSEHOLD_WAGE_TERCILES,
                  "woman_education": EDUCATION_LEVELS}

# Paper, Table 1 / A1, Denmark: shares of couple-years and crude first-birth rates.
PAPER_SHARES = {
    "woman_age_group": {"18-25": .368, "26-30": .311, "31-35": .145, "36-45": .175},
    "earner_constellation": {
        "both_jobless": .085, "male_works": .1742, "female_works": .1127,
        "dual_equal_earners": .2931, "dual_male_main_earner": .2362,
        "dual_female_main_earner": .0988},
    "woman_education": {"enrolled": .013, "low": .588, "high": .378, "unknown": .021},
    "either_partner_foreign": {1: .194},
}
PAPER_FIRST_BIRTH_RATES = {
    "woman_age_group": {"18-25": .08, "26-30": .14, "31-35": .14, "36-45": .04},
    "earner_constellation": {
        "both_jobless": .06, "male_works": .10, "female_works": .08,
        "dual_equal_earners": .16, "dual_male_main_earner": .11,
        "dual_female_main_earner": .12},
}
# Approximate Danish Model 1 marginal effects: the paper's Table A2 Model 1
# coefficients for Denmark, evaluated on a composition matched to Table A1
# (consistent with the Denmark bars in the paper's Figure 1, which gives no
# exact numbers).
PAPER_MARGINAL_EFFECTS_MODEL1 = {
    "dual_equal_earners": 0.0, "both_jobless": -.096, "male_works": -.059,
    "female_works": -.071, "dual_male_main_earner": -.043,
    "dual_female_main_earner": -.034}


def load_cells():
    cells = pd.read_csv(EVENT_TABLE_FILE)
    for variable, categories in CATEGORY_ORDER.items():
        cells[variable] = pd.Categorical(cells[variable], categories=categories)
    return cells


def fit_model(cells, model_name):
    """Discrete-time logit hazard, fitted as a binomial model on the cells:
    first births out of couple-years. This gives exactly the same estimates as
    a logit on the individual couple-years."""
    design_matrix = dmatrix(MODEL_FORMULAS[model_name], cells, return_type="dataframe")
    births_and_non_births = np.column_stack([cells.first_births,
                                             cells.couple_years - cells.first_births])
    fitted = sm.GLM(births_and_non_births, design_matrix, family=sm.families.Binomial()).fit()
    return fitted, design_matrix.design_info


def marginal_effects(cells, fitted, design_info):
    """Average marginal effect of each constellation against dual equal earners.

    The effect is the change in the yearly probability of a conception leading
    to a first birth if every couple-year had this constellation instead of
    dual equal earners, with all other traits kept as observed.
    """
    cell_weight = cells.couple_years.to_numpy() / cells.couple_years.sum()
    coefficients = fitted.params.to_numpy()
    coefficient_covariance = fitted.cov_params().to_numpy()

    def average_prediction_if_all_were(constellation_name):
        # Give every cell this constellation, predict, and average over the cells.
        counterfactual = cells.copy()
        counterfactual["earner_constellation"] = pd.Categorical(
            [constellation_name] * len(cells), categories=CONSTELLATIONS)
        design_matrix = np.asarray(build_design_matrices([design_info], counterfactual)[0])
        probability = expit(design_matrix @ coefficients)
        average_probability = cell_weight @ probability
        # Gradient of that average with respect to the coefficients, for the standard error.
        gradient = (cell_weight * probability * (1 - probability)) @ design_matrix
        return average_probability, gradient

    reference_probability, reference_gradient = average_prediction_if_all_were("dual_equal_earners")
    rows = []
    for constellation_name in CONSTELLATIONS:
        probability, gradient = average_prediction_if_all_were(constellation_name)
        # Delta method: the variance of a difference of predictions is
        # gradient' x covariance x gradient.
        gradient_of_difference = gradient - reference_gradient
        rows.append({
            "earner_constellation": constellation_name,
            "average_probability": probability,
            "marginal_effect": probability - reference_probability,
            "standard_error": float(np.sqrt(
                gradient_of_difference @ coefficient_covariance @ gradient_of_difference))})
    return pd.DataFrame(rows)


def true_marginal_effects(cells):
    """The marginal effects implied by stage 1's birth probabilities, evaluated
    on the same cells. This is what Model 2 should recover."""
    cell_weight = cells.couple_years.to_numpy() / cells.couple_years.sum()
    average_probability = np.array([
        cell_weight @ stage1_simulate_registers.birth_probability(
            cells.woman_age_group.astype(str), [constellation_name] * len(cells),
            cells.woman_education.astype(str), cells.household_wage_tercile.astype(str),
            cells.either_partner_foreign)
        for constellation_name in CONSTELLATIONS])
    # CONSTELLATIONS[0] is the reference, dual equal earners.
    return pd.DataFrame({"earner_constellation": CONSTELLATIONS,
                         "average_probability": average_probability,
                         "marginal_effect": average_probability - average_probability[0]})


def composition_compared_with_paper(cells):
    """Share of couple-years and crude first-birth rate of each category,
    next to the paper's Danish Table A1."""
    rows = []
    all_couple_years = cells.couple_years.sum()
    for variable, paper_shares in PAPER_SHARES.items():
        totals = cells.groupby(variable, observed=True)[["first_births", "couple_years"]].sum()
        for category, paper_share in paper_shares.items():
            share = totals.couple_years.get(category, 0) / all_couple_years
            first_birth_rate = (totals.first_births.get(category, 0)
                                / max(totals.couple_years.get(category, 1), 1))
            paper_rate = PAPER_FIRST_BIRTH_RATES.get(variable, {}).get(category, np.nan)
            rows.append([variable, category, paper_share, share, paper_rate, first_birth_rate])
    return pd.DataFrame(rows, columns=["variable", "category", "share_in_paper", "share_simulated",
                                       "first_birth_rate_in_paper", "first_birth_rate_simulated"])


def plot_marginal_effects(effects, figure_file):
    """Bars for Model 1 and Model 2 with 95% confidence intervals; white
    diamonds for the stage 1 truth."""
    shown = CONSTELLATIONS[1:]                    # all but the reference
    position = np.arange(len(shown))
    bar_width = 0.36
    figure, axis = plt.subplots(figsize=(9, 4.8), dpi=150)
    for i, (model_name, colour, label) in enumerate((
            ("model1", "#2a78d6", "Model 1: age only"),
            ("model2", "#eb6834", "Model 2: + controls"))):
        model = effects[model_name].set_index("earner_constellation").loc[shown]
        bar_position = position + (i - 0.5) * (bar_width + 0.02)
        axis.bar(bar_position, model.marginal_effect, bar_width, color=colour, label=label, zorder=2)
        axis.errorbar(bar_position, model.marginal_effect, yerr=1.96 * model.standard_error,
                      fmt="none", ecolor="#1f1f1e", lw=1, capsize=2, zorder=3)
    truth = effects["truth"].set_index("earner_constellation").loc[shown]
    axis.scatter(position + 0.5 * (bar_width + 0.02), truth.marginal_effect, marker="D", s=34,
                 facecolor="white", edgecolor="#1f1f1e", lw=1.3, zorder=4,
                 label="Truth set in stage 1 (Model 2 scale)")
    axis.axhline(0, color="#6b6a63", lw=0.8)
    axis.set_xticks(position, [CONSTELLATION_LABELS[c] for c in shown], fontsize=9)
    axis.set_ylabel("AME on P(conception → first birth)", fontsize=9)
    axis.set_title("Simulated Denmark: AMEs vs. dual earners with equal wages", fontsize=11, loc="left")
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

    # Fit both models; keep their marginal effects and odds ratios.
    effects, odds_ratio_tables = {}, []
    for model_name in ("model1", "model2"):
        fitted, design_info = fit_model(cells, model_name)
        effects[model_name] = marginal_effects(cells, fitted, design_info)
        odds_ratios = pd.DataFrame({"odds_ratio": np.exp(fitted.params), "p_value": fitted.pvalues})
        odds_ratios["model"] = model_name
        odds_ratio_tables.append(odds_ratios.rename_axis("term").reset_index())
    effects["truth"] = true_marginal_effects(cells)
    pd.concat(odds_ratio_tables).to_csv(OUTPUT_FOLDER / "4_odds_ratios.csv", index=False)

    # One row per constellation: both models, the truth, and the paper.
    summary = pd.DataFrame({
        "constellation": [CONSTELLATION_LABELS[c] for c in CONSTELLATIONS],
        "marginal_effect_model1": effects["model1"].marginal_effect,
        "standard_error_model1": effects["model1"].standard_error,
        "marginal_effect_model2": effects["model2"].marginal_effect,
        "standard_error_model2": effects["model2"].standard_error,
        "true_marginal_effect": effects["truth"].marginal_effect,
        "paper_marginal_effect_model1": [PAPER_MARGINAL_EFFECTS_MODEL1[c] for c in CONSTELLATIONS],
    })
    summary.to_csv(OUTPUT_FOLDER / "4_marginal_effects.csv", index=False)
    composition = composition_compared_with_paper(cells)
    composition.to_csv(OUTPUT_FOLDER / "4_composition_vs_paper.csv", index=False)
    plot_marginal_effects(effects, OUTPUT_FOLDER / "4_fig_marginal_effects.png")

    print(f"  probability for dual equal earners: "
          f"model1 {effects['model1'].average_probability[0]:.3f} (paper .157), "
          f"model2 {effects['model2'].average_probability[0]:.3f} (paper .125)")
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string(index=False))
    print(composition.round(3).to_string(index=False))
    print(f"  outputs: {OUTPUT_FOLDER}/4_*")


if __name__ == "__main__":
    print("Stage 4: models and comparison ...")
    analyse()
