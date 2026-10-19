"""
STAGE 2 - Process the raw registers into the data the model is fitted on, in
two steps:

  1. Build the couple-year file: one row per couple per calendar year in which
     the woman is at risk of a first birth. This uses ONLY the stage 1 register
     files in 1simulated_register_data/, as a researcher at Statistics Denmark
     would, and follows the paper's sample rules. Every register row describes
     a person's situation at the END of the calendar year.
  2. Collapse the couple-year file into the event-occurrence table: one row per
     combination of the model variables, with the number of first births and
     the number of couple-years. This is the format the paper's authors
     received from each country team.
"""
import numpy as np
import pandas as pd

from common import (COUPLE_YEAR_FILE, EVENT_TABLE_FILE, MAX_AGE_AT_RISK, MIN_AGE_AT_RISK,
                    PROCESSED_DATA_FOLDER, REGISTER_FILES, YEARS, age_group, constellation,
                    wage_tercile)

# The model variables that define a cell of the event-occurrence table.
CELL_VARIABLES = ["woman_age_group", "earner_constellation", "household_wage_tercile",
                  "woman_education", "either_partner_foreign"]


def build_couple_years():
    population = pd.read_csv(REGISTER_FILES["bef"])    # who, age, origin, partner
    employment = pd.read_csv(REGISTER_FILES["ras"])    # employed or not, per person-year
    income = pd.read_csv(REGISTER_FILES["ind"])        # labour income, per person-year
    education = pd.read_csv(REGISTER_FILES["udda"])    # enrolment and degree, per person-year
    births = pd.read_csv(REGISTER_FILES["fod"], parse_dates=["birth_date"])

    # ---- who is in the sample ---------------------------------------------------
    # Start from women in the observation years.
    women = population[(population.sex == "F") & population.year.isin(YEARS)].copy()
    # Keep the years in which she has a co-resident partner at year end.
    women = women[women.partner_id > 0]
    # Keep the years in which she is aged 18-45.
    women["woman_age"] = women.year - women.birth_year
    women = women[women.woman_age.between(MIN_AGE_AT_RISK, MAX_AGE_AT_RISK)]

    # ---- childlessness and the event ----------------------------------------------
    # The year of her first birth: her earliest birth date in the birth
    # register, searched over ALL years and not just the observation years.
    # Empty if she has no child.
    first_birth_year = (births.groupby("mother_id")["birth_date"].min().dt.year
                        .rename("first_birth_year"))
    women = women.join(first_birth_year, on="person_id")
    # Keep a year only if she is still at risk: no first birth, or a first
    # birth in a later year, so she is childless through the end of the year.
    women = women[women.first_birth_year.isna() | (women.first_birth_year > women.year)]
    # The event: her first child is born the next year. This is the paper's
    # "conception leading to a first birth", dated to the year before the
    # birth. It is assigned to the couple of that year. After her event year a
    # woman never appears in the file again.
    women["conceived_first_child_that_year"] = (women.first_birth_year == women.year + 1).astype(int)

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
        women.woman_employed, women.man_employed, women.woman_wage, women.man_wage)
    # Household wages, and their terciles among the couple-years of the same
    # calendar year (see common.wage_tercile). Stage 1 uses the same rule.
    women["household_wage"] = women.woman_wage + women.man_wage
    women["household_wage_tercile"] = women.groupby("year")["household_wage"].transform(wage_tercile)
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
    # A couple is the pair (woman_id, man_id).
    couple_years = women.rename(columns={"person_id": "woman_id", "partner_id": "man_id"})[[
        "woman_id", "man_id", "year", "woman_age", "woman_age_group",
        "woman_employed", "man_employed", "woman_wage", "man_wage", "earner_constellation",
        "household_wage", "household_wage_tercile", "woman_education",
        "either_partner_foreign", "conceived_first_child_that_year"]]
    couple_years = couple_years.sort_values(["woman_id", "year"]).reset_index(drop=True)
    PROCESSED_DATA_FOLDER.mkdir(exist_ok=True)
    couple_years.to_csv(COUPLE_YEAR_FILE, index=False)

    number_of_couples = len(couple_years[["woman_id", "man_id"]].drop_duplicates())
    print(f"  {COUPLE_YEAR_FILE.name}: {len(couple_years):,} couple-years, "
          f"{number_of_couples:,} couples "
          f"({len(couple_years) / number_of_couples:.2f} years per couple), "
          f"{couple_years.conceived_first_child_that_year.sum():,} first births")
    return couple_years


def count_events(couple_years):
    """Collapse the couple-year file into the event-occurrence table."""
    # Group the couple-years by every model variable. That is at most
    # 4 x 6 x 3 x 4 x 2 = 576 cells; combinations that never occur produce no
    # row. For each cell: first_births = number of couple-years with the event,
    # couple_years = number of rows (the exposure). The crude rate of a cell is
    # first_births / couple_years, and a logit fitted to (first_births out of
    # couple_years) gives exactly the same estimates as a logit on the
    # individual couple-years.
    cells = (couple_years.groupby(CELL_VARIABLES)["conceived_first_child_that_year"]
             .agg(first_births="sum", couple_years="size").reset_index())
    cells.to_csv(EVENT_TABLE_FILE, index=False)
    print(f"  {EVENT_TABLE_FILE.name}: {len(cells)} cells (of max 576)")
    return cells


def process():
    """Both steps: registers -> couple-years -> event-occurrence table."""
    return count_events(build_couple_years())


if __name__ == "__main__":
    print("Stage 2: processing the registers ...")
    process()
