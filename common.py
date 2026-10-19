"""Definitions shared by all stages: file locations, category labels, and the
three classification rules (age group, wage tercile, earner constellation)."""
from pathlib import Path

import numpy as np
import pandas as pd

# One folder per kind of data, in the order the pipeline produces them.
PROJECT_FOLDER = Path(__file__).parent
DST_DATA_FOLDER = PROJECT_FOLDER / "0data"                        # stage 0: Statistics Denmark figures
SIMULATED_REGISTER_FOLDER = PROJECT_FOLDER / "1simulated_register_data"   # stage 1: simulated registers
PROCESSED_DATA_FOLDER = PROJECT_FOLDER / "2processed_data"            # stage 2: couple-years, counts
ESTIMATION_FOLDER = PROJECT_FOLDER / "3estimation"                    # stage 3: estimation results

# The five simulated registers, named after the Danish registers they imitate:
# bef = population, ras = employment, ind = income, udda = education, fod = births.
REGISTER_FILES = {name: SIMULATED_REGISTER_FOLDER / f"{name}.csv"
                  for name in ("bef", "ras", "ind", "udda", "fod")}
COUPLE_YEAR_FILE = PROCESSED_DATA_FOLDER / "couple_years.csv"
EVENT_TABLE_FILE = PROCESSED_DATA_FOLDER / "event_occurrence_table.csv"

# The paper's observation period. Stage 0 downloads the yearly figures for these
# years, stage 1 simulates them, and stage 2 turns them into couple-years.
YEARS = range(2010, 2020)

# The women at risk of a first birth: aged 18-45 (year - birth year), as in
# the paper. Stage 1 simulates conceptions for them, stage 2 selects them.
MIN_AGE_AT_RISK, MAX_AGE_AT_RISK = 18, 45
# The ages covered by the yearly Statistics Denmark table. People outside this
# range get the figures of the nearest age in it.
YOUNGEST_AGE, OLDEST_AGE = 18, 65

# Category lists. The first entry of each is the reference category in the model.
AGE_GROUPS = ["18-25", "26-30", "31-35", "36-45"]
CONSTELLATIONS = ["dual_equal_earners", "both_jobless", "male_works", "female_works",
                  "dual_male_main_earner", "dual_female_main_earner"]
# The paper's labels (BW = breadwinner, ME = main earner).
CONSTELLATION_LABELS = {
    "dual_equal_earners": "Dual BW, equal earners", "both_jobless": "Jobless couple",
    "male_works": "Male BW", "female_works": "Female BW",
    "dual_male_main_earner": "Dual BW, male ME", "dual_female_main_earner": "Dual BW, female ME"}
HOUSEHOLD_WAGE_TERCILES = ["low", "middle", "high"]

# The paper's Table A2, Model 2, for Denmark: a ready-made input, worked out
# once from the paper and not regenerated. Stage 1 generates the births from
# it and stage 3 checks the estimates against it.
# Table A2 uses NORWAY as the reference country, so it prints no Danish odds
# ratio directly. Each Danish odds ratio is its main row times its
# "Denmark * ..." row (jobless couples: 0.670 x 1.132 = 0.758), and the Danish
# baseline odds are Constant x Denmark (0.133 x 0.581 = 0.0773). The woman's
# age has no "Denmark * age" row, so its main row is used as it stands.
# The baseline is the odds of the reference couple: woman aged 18-25, dual
# equal earners, low household wages, low education, neither partner foreign.
TABLE_A2_FILE = DST_DATA_FOLDER / "paper_table_a2_model2_denmark.csv"


def load_danish_table_a2():
    """The Danish baseline odds, and the odds ratios as
    {variable: {category: odds ratio}}, reference category first."""
    table = pd.read_csv(TABLE_A2_FILE, dtype={"category": str}, keep_default_na=False)
    baseline_odds = table.loc[table.variable == "baseline", "danish_odds_ratio"].item()
    odds_ratios = {}
    for row in table[table.variable != "baseline"].itertuples():
        # Foreign origin is a 0/1 number in the data, not a label.
        category = int(row.category) if row.variable == "either_partner_foreign" else row.category
        odds_ratios.setdefault(row.variable, {})[category] = row.danish_odds_ratio
    return baseline_odds, odds_ratios


DANISH_BASELINE_ODDS, DANISH_ODDS_RATIOS = load_danish_table_a2()


def age_group(woman_age):
    """The paper's four age groups of the woman: 18-25, 26-30, 31-35, 36-45."""
    woman_age = np.asarray(woman_age)
    return np.select([woman_age <= 25, woman_age <= 30, woman_age <= 35],
                     AGE_GROUPS[:3], AGE_GROUPS[3])


def wage_tercile(household_wage):
    """Low, middle or high third of the household wages passed in. Stages 1 and 2
    both apply this to the couples at risk in one calendar year, so that wage
    growth over the years does not move couples between thirds."""
    lower_limit, upper_limit = np.quantile(household_wage, [1 / 3, 2 / 3])
    return np.array(HOUSEHOLD_WAGE_TERCILES)[np.digitize(household_wage, [lower_limit, upper_limit])]


def constellation(woman_employed, man_employed, woman_wage, man_wage,
                  equal_share_from=0.40, equal_share_to=0.60):
    """The paper's six employment-wage constellations of a couple."""
    woman_employed = np.asarray(woman_employed, bool)
    man_employed = np.asarray(man_employed, bool)
    woman_wage = np.asarray(woman_wage, float)
    man_wage = np.asarray(man_wage, float)

    # Her share of the couple's combined wages. If both wages are 0 the share
    # is set to 0.5 (equal); this doesn't occur with the default parameters.
    combined_wage = woman_wage + man_wage
    woman_wage_share = np.divide(woman_wage, combined_wage,
                                 out=np.full_like(combined_wage, 0.5), where=combined_wage > 0)

    # Employment status decides first; the first matching condition wins.
    # Wages only split the couples where both work: a "male works" couple
    # stays in that category even if she has some part-year income. "Jobless"
    # means not employed (students, inactive, unemployed), as in the paper.
    # A share of exactly 0.40 or 0.60 counts as equal.
    both_employed = woman_employed & man_employed
    return np.select(
        [~woman_employed & ~man_employed,                        # neither employed
         man_employed & ~woman_employed,                         # only he is employed
         woman_employed & ~man_employed,                         # only she is employed
         both_employed & (woman_wage_share < equal_share_from),  # she earns under 40%
         both_employed & (woman_wage_share > equal_share_to)],   # she earns over 60%
        ["both_jobless", "male_works", "female_works",
         "dual_male_main_earner", "dual_female_main_earner"],
        "dual_equal_earners")                                    # she earns 40-60%
