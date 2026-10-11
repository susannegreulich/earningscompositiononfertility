"""Definitions shared by all stages: file locations, category labels, and the
three classification rules (age group, wage tercile, earner constellation)."""
from pathlib import Path

import numpy as np

# One folder per kind of data, in the order the pipeline produces them.
PROJECT_FOLDER = Path(__file__).parent
DST_DATA_FOLDER = PROJECT_FOLDER / "0dst_data"                        # stage 0: Statistics Denmark figures
SIMULATED_REGISTER_FOLDER = PROJECT_FOLDER / "1simulated_register_data"   # stage 1: simulated registers
PROCESSED_DATA_FOLDER = PROJECT_FOLDER / "2processed_data"            # stage 2: couple-years, counts
ESTIMATION_FOLDER = PROJECT_FOLDER / "3estimation"                    # stage 3: estimation results

# The five simulated registers, named after the Danish registers they imitate:
# bef = population, ras = employment, ind = income, udda = education, fod = births.
REGISTER_FILES = {name: SIMULATED_REGISTER_FOLDER / f"{name}.csv"
                  for name in ("bef", "ras", "ind", "udda", "fod")}
COUPLE_YEAR_FILE = PROCESSED_DATA_FOLDER / "couple_years.csv"
EVENT_TABLE_FILE = PROCESSED_DATA_FOLDER / "event_occurrence_table.csv"

# Category lists. The first entry of each is the reference category in the model.
AGE_GROUPS = ["18-25", "26-30", "31-35", "36-45"]
CONSTELLATIONS = ["dual_equal_earners", "both_jobless", "male_works", "female_works",
                  "dual_male_main_earner", "dual_female_main_earner"]
# The paper's labels (BW = breadwinner, ME = main earner).
CONSTELLATION_LABELS = {
    "dual_equal_earners": "Dual BW, equal earners", "both_jobless": "Jobless couple",
    "male_works": "Male BW", "female_works": "Female BW",
    "dual_male_main_earner": "Dual BW, male ME", "dual_female_main_earner": "Dual BW, female ME"}
EDUCATION_LEVELS = ["low", "enrolled", "high", "unknown"]
HOUSEHOLD_WAGE_TERCILES = ["low", "middle", "high"]


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
