"""Definitions shared by all stages: file locations, category labels, and the
two classification rules (age group, earner constellation)."""
from pathlib import Path

import numpy as np

# Every stage writes into one folder; the number at the start of a file name
# is the stage that wrote it.
OUTPUT_FOLDER = Path(__file__).parent / "output"
# The five simulated registers, named after the Danish registers they imitate:
# bef = population, ras = employment, ind = income, udda = education, fod = births.
REGISTER_FILES = {name: OUTPUT_FOLDER / f"1_simulated_register_{name}.csv"
                  for name in ("bef", "ras", "ind", "udda", "fod")}
COUPLE_YEAR_FILE = OUTPUT_FOLDER / "2_couple_years.csv"
EVENT_TABLE_FILE = OUTPUT_FOLDER / "3_event_occurrence_table.csv"

# Category lists. The first entry of each is the reference category in the models.
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
