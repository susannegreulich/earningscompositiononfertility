"""
STAGE 2 - Build the couple-year file from the raw registers: one row per
couple per calendar year in which the woman is at risk of a first birth.

Uses ONLY the stage 1 register files, as a researcher at Statistics Denmark
would, and follows the paper's sample rules. Every register row describes a
person's situation at the END of the calendar year.
"""
import numpy as np
import pandas as pd

from common import COUPLE_YEAR_FILE, REGISTER_FILES, age_group, constellation

# =============================================================================
# PARAMETERS - sample definition (paper: Denmark 2010-2019, women 18-45)
# =============================================================================
OBSERVATION_YEARS = range(2010, 2020)   # calendar years that become couple-years
MIN_AGE, MAX_AGE = 18, 45               # woman's age (year - birth year)
YEARS_FROM_EVENT_TO_BIRTH = 1           # the event is a first birth this many years later
EQUAL_EARNER_SHARE = (0.40, 0.60)       # her wage share counted as "equal earners"
# =============================================================================


def build():
    population = pd.read_csv(REGISTER_FILES["bef"])    # who, age, origin, partner
    employment = pd.read_csv(REGISTER_FILES["ras"])    # employed or not, per person-year
    income = pd.read_csv(REGISTER_FILES["ind"])        # labour income, per person-year
    education = pd.read_csv(REGISTER_FILES["udda"])    # enrolment and degree, per person-year
    births = pd.read_csv(REGISTER_FILES["fod"], parse_dates=["birth_date"])

    # ---- who is in the sample ---------------------------------------------------
    # Start from women in the observation years.
    women = population[(population.sex == "F") & population.year.isin(OBSERVATION_YEARS)].copy()
    # Keep the years in which she has a co-resident partner at year end.
    women = women[women.partner_id > 0]
    # Keep the years in which she is aged 18-45.
    women["woman_age"] = women.year - women.birth_year
    women = women[women.woman_age.between(MIN_AGE, MAX_AGE)]

    # ---- childlessness and the event ----------------------------------------------
    # The year of her first birth: her earliest birth date in the birth
    # register, searched over ALL years and not just the observation years.
    # Empty if she has no child.
    first_birth_year = (births.groupby("mother_id")["birth_date"].min().dt.year
                        .rename("first_birth_year"))
    women = women.join(first_birth_year, on="person_id")
    # Keep a year only if she is still at risk: no first birth, or a first
    # birth no earlier than `year + YEARS_FROM_EVENT_TO_BIRTH`. With the
    # default of 1: childless through the end of the year.
    birth_year_of_event = women.year + YEARS_FROM_EVENT_TO_BIRTH
    women = women[women.first_birth_year.isna() | (women.first_birth_year >= birth_year_of_event)]
    # The event: her first child is born exactly YEARS_FROM_EVENT_TO_BIRTH
    # years later. This is the paper's "conception leading to a first birth",
    # dated to the year before the birth. It is assigned to the couple of that
    # year, even if the union ends before the birth. After her event year a
    # woman never appears in the file again.
    women["conceived_first_child_that_year"] = (
        women.first_birth_year == women.year + YEARS_FROM_EVENT_TO_BIRTH).astype(int)

    # ---- attaching characteristics --------------------------------------------------
    person_year = ["person_id", "year"]
    # Her employment status, labour income and education in the same year.
    women = women.merge(employment.rename(columns={"employed": "woman_employed"}),
                        on=person_year, how="left")
    women = women.merge(income.rename(columns={"wage_dkk": "woman_wage"}),
                        on=person_year, how="left")
    women = women.merge(education, on=person_year, how="left")
    # His origin, employment status and labour income in the same year, found
    # through her partner_id.
    men = (population[population.sex == "M"][["person_id", "year", "origin"]]
           .merge(employment, on=person_year).merge(income, on=person_year)
           .rename(columns={"person_id": "partner_id", "origin": "man_origin",
                            "employed": "man_employed", "wage_dkk": "man_wage"}))
    women = women.merge(men, on=["partner_id", "year"], how="left")
    # Drop the couple-year if either partner's employment or wage is missing
    # (paper: complete information required). Missing education is NOT dropped;
    # it becomes the category "unknown" below.
    women = women.dropna(subset=["woman_employed", "man_employed", "woman_wage", "man_wage"])

    # ---- derived variables -------------------------------------------------------------
    # Employment-wage constellation: who works and, if both do, how wages are
    # split (see common.constellation).
    women["earner_constellation"] = constellation(
        women.woman_employed, women.man_employed, women.woman_wage, women.man_wage,
        *EQUAL_EARNER_SHARE)
    # Household wages, and their terciles across ALL couple-years in the file,
    # pooled over years. Ties are broken by row order, so each tercile holds
    # exactly one third. (Stage 1 instead uses terciles computed each year
    # among the couples at risk, which assigns about 2% of couple-years
    # differently.)
    women["household_wage"] = women.woman_wage + women.man_wage
    women["household_wage_tercile"] = pd.qcut(
        women.household_wage.rank(method="first"), 3, labels=["low", "middle", "high"]).astype(str)
    # Her education category; the first matching condition wins:
    # currently enrolled -> "enrolled"; degree missing -> "unknown";
    # ISCED 5 or above -> "high"; otherwise "low".
    women["woman_education"] = np.select(
        [women.enrolled == 1, women.isced_level.isna(), women.isced_level >= 5],
        ["enrolled", "unknown", "high"], "low")
    # Foreign origin: either partner born abroad.
    women["either_partner_foreign"] = (
        (women.origin == "foreign") | (women.man_origin == "foreign")).astype(int)
    women["woman_age_group"] = age_group(women.woman_age)

    # ---- write the file --------------------------------------------------------------------
    # A couple is the pair (woman_id, man_id): if she moves in with a new
    # partner, that is a new couple.
    couple_years = women.rename(columns={"person_id": "woman_id", "partner_id": "man_id"})[[
        "woman_id", "man_id", "year", "woman_age", "woman_age_group",
        "woman_employed", "man_employed", "woman_wage", "man_wage", "earner_constellation",
        "household_wage", "household_wage_tercile", "woman_education",
        "either_partner_foreign", "conceived_first_child_that_year"]]
    couple_years = couple_years.sort_values(["woman_id", "year"]).reset_index(drop=True)
    couple_years.to_csv(COUPLE_YEAR_FILE, index=False)

    number_of_couples = len(couple_years[["woman_id", "man_id"]].drop_duplicates())
    print(f"  {COUPLE_YEAR_FILE.name}: {len(couple_years):,} couple-years, "
          f"{number_of_couples:,} couples "
          f"({len(couple_years) / number_of_couples:.2f} years per couple; paper DK 2.95), "
          f"{couple_years.conceived_first_child_that_year.sum():,} first births")
    return couple_years


if __name__ == "__main__":
    print("Stage 2: building couple-year file ...")
    build()
