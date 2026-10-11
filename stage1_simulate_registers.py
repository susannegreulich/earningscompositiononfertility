"""
STAGE 1 - Simulate raw, person-level registers (the data that never leaves
Statistics Denmark in the real study).

Who is simulated: the women born 1965-2001 who were childless at the start of
2010, and the partners they live with. They are followed through 2010-2019.

How their traits are set:
  - drawn once and kept: birth year, origin, education, and when (if ever) she
    moves in with a partner;
  - drawn every year: employed, enrolled, wage. Each is drawn from that year's
    Statistics Denmark average for the person's sex and age, and does not
    depend on any other trait; however a degree of persistence is built-in to avoid jumps
  - the first conceptions/births are generated using the paper's model and estimates: whether a childless woman with a
    partner conceives a first child that year.

The Statistics Denmark figures are read from 0dst_data/, which stage 0
(stage0_download_dst_data.py) downloads; the comments there name each table.
The partner age gap is also set in stage 0. The only numbers typed into this
file are the paper's odds ratios and one assumption, P_KEEP_LAST_YEARS_POSITION.
"""
import numpy as np
import pandas as pd

from common import (DST_DATA_FOLDER, REGISTER_FILES, SIMULATED_REGISTER_FOLDER, age_group,
                    constellation, wage_tercile)
from stage0_download_dst_data import (DANISH_BASELINE_ODDS, DANISH_ODDS_RATIOS,
                                      PARTNER_AGE_GAP_MEAN_AND_SD)

# =============================================================================
# PARAMETERS - edit these
# =============================================================================
SEED = 2026
# Share of Denmark's childless women that is simulated. 1.0 is every one of
# them (about 760,000 women, which takes several minutes and over 1 GB of files).
POPULATION_SCALE = 0.2
BIRTH_COHORTS = range(1965, 2002)   # birth years of the simulated women
YEARS = range(2010, 2020)           # years written to the register files

# ---- STAYING THE SAME FROM YEAR TO YEAR -----------------------------------------
# Employment, enrolment and wage are drawn from each year's Statistics Denmark
# figures. To stop a person's status jumping at random between years, each
# person holds a position in the distribution (for example "at the 30th
# percentile of wages") and keeps it from one year to the next with this
# probability; otherwise a new position is drawn. 0 = a fresh draw every year,
# 1 = the same position for life. The share of people employed, enrolled or at
# a given wage in any one year is the Statistics Denmark figure whatever this
# value is. Assumption: Statistics Denmark does not publish it.
P_KEEP_LAST_YEARS_POSITION = 0.8

# ---- FIRST BIRTH ------------------------------------------------------------
# Births are generated from the paper's logit model with controls (eq. 1 with
# added controls, i.e. Model 2):
#
#   logit P(conception leading to a first birth this year)
#       = ln(baseline odds) + the sum of ln(odds ratio) of each of the couple's traits
#
# The baseline odds and the odds ratios are the DANISH values of the paper's
# Table A2. They are typed in and computed in stage 0 (see the section "THE
# PAPER'S TABLE A2" there), and listed in 0dst_data/paper_table_a2_model2_denmark.csv.
# To try other values, change them in stage 0.
FIRST_BIRTH_BASELINE_ODDS = DANISH_BASELINE_ODDS
FIRST_BIRTH_ODDS_RATIOS = DANISH_ODDS_RATIOS
# =============================================================================


def load_yearly_rates():
    """The yearly Statistics Denmark figures, as a lookup: rate(name, sex, year, age)."""
    table = pd.read_csv(DST_DATA_FOLDER / "yearly_rates.csv")
    youngest_age, first_year = table.age.min(), table.year.min()
    by_sex = {sex: one_sex.set_index(["year", "age"]) for sex, one_sex in table.groupby("sex")}

    def rate(name, sex, year, age):
        # Ages outside the table (18-65) get the figure of the nearest age in it.
        figures = by_sex[sex][name].unstack("age").to_numpy()
        return figures[year - first_year, np.clip(age - youngest_age, 0, figures.shape[1] - 1)]
    return rate


def birth_probability(**traits):
    """The yearly probability that a couple conceives a first child.

    Takes the couple's traits and returns a probability. It is the probability
    for the COUPLE, not only the woman: her age group and education are hers,
    but the earner constellation, the household-wage tercile and foreign origin
    describe both partners.

    It works on many couples at once: each input is an array with one entry per
    couple (one array per variable in FIRST_BIRTH_ODDS_RATIOS, for example
    earner_constellation=[...]), and the result is one probability per couple.

    Example for one couple: she is 28 with a higher degree, both work with equal
    wages, household wages are in the middle third, neither was born abroad.
        baseline odds (0.133 x 0.581)             0.0773
        x age 26-30                               1.324
        x dual equal earners (reference)          1.0
        x middle wage tercile (1.349 x 1.277)     1.723
        x high education (1.197 x 1.141)          1.366
        x not foreign (reference)                 1.0
        = odds                                    0.241
        probability = odds / (1 + odds)           0.194
    So this couple has about a 19% chance of conceiving a first child that year.

    This function only returns the probability. The yes or no is decided in
    yearly_traits(), which draws a random number between 0 and 1 for each couple; a
    conception occurs if it falls below the probability.
    """
    # The same calculation on the log scale: add up ln(odds ratio) for each
    # trait, then turn the sum into a probability.
    log_odds = np.log(FIRST_BIRTH_BASELINE_ODDS)
    for variable, odds_ratios in FIRST_BIRTH_ODDS_RATIOS.items():
        log_odds = log_odds + np.log([odds_ratios[value] for value in traits[variable]])
    return 1 / (1 + np.exp(-log_odds))


def new_positions(random, count):
    """Where a person stands in each yearly distribution: a number between 0
    and 1 for employment and enrolment, a standard normal draw for wages."""
    return {"employment": random.random(count), "enrolment": random.random(count),
            "wage": random.normal(0, 1, count)}


def traits_this_year(rate, sex, year, age, position, education_position, high_education):
    """One year's employment, enrolment, education and wage, from the Statistics
    Denmark figures for the person's sex and age. Nothing is drawn here: each
    person's position is compared with the published rate."""
    # Employed if the position is below the employment rate (RAS200); enrolled
    # if below the enrolment rate (UDDAKT10 / FOLK1E).
    employed = position["employment"] < rate("employment_rate", sex, year, age)
    enrolled = position["enrolment"] < rate("enrolment_rate", sex, year, age)
    # Education (HFUDD11): high if the fixed position is below the share with
    # a higher degree; unknown if it is in the top share with education not stated.
    high_education |= education_position < rate("share_high_education", sex, year, age)
    education_unknown = education_position > 1 - rate("share_education_unknown", sex, year, age)
    # Wage: log-normal, with the average of people with wage income
    # (INDKP111) and the spread of hourly earnings (LONS50). 0 if not employed.
    log_sd = rate("wage_log_sd", sex, year, age)
    log_mean = np.log(rate("average_wage", sex, year, age)) - log_sd ** 2 / 2
    wage = np.where(employed, np.round(np.exp(log_mean + log_sd * position["wage"]), -2), 0.0)
    return employed, enrolled & ~education_unknown, education_unknown, wage


def fixed_traits(random):
    """PART 1 - The women, and everything about them that is drawn once and
    never changes: how many there are, birth year, origin, education position,
    and the year each moves in with a partner.

    Every woman is one position in each of the arrays returned.
    """
    women_2010 = pd.read_csv(DST_DATA_FOLDER / "women_2010.csv").set_index("age_on_1_january")
    immigrant_share = (pd.read_csv(DST_DATA_FOLDER / "immigrant_share_2010.csv")
                       .set_index(["sex", "age"])["share_immigrant"])
    first_year = min(YEARS)

    # ---- who is simulated ---------------------------------------------------------
    # The count of childless women at the start of 2010, per birth year (BARNLOS
    # and FOLK1E), times POPULATION_SCALE. Women who were already mothers are
    # left out: they can never have a first birth.
    cohorts = women_2010[women_2010.birth_year.isin(BIRTH_COHORTS)]
    women_per_cohort = (cohorts.childless_women * POPULATION_SCALE).round().astype(int)
    woman_birth_year = np.repeat(cohorts.birth_year.to_numpy(), women_per_cohort)
    number_of_women = len(woman_birth_year)
    woman_age_at_start = first_year - 1 - woman_birth_year         # age on 1 January 2010

    # ---- origin and education --------------------------------------------------------
    # Born abroad: the share of immigrants among women of her age in 2010 (FOLK1E).
    woman_foreign_born = random.random(number_of_women) < immigrant_share["F"].reindex(
        woman_age_at_start).to_numpy()
    # Education: her fixed position among women of her age. Each year she counts
    # as high-educated if that position is below the share with a higher degree
    # at her age (HFUDD11); once reached, the degree is kept.
    woman_education_position = random.random(number_of_women)

    # ---- when she moves in with a partner (drawn once) --------------------------------
    # At the start of 2010 she lives with a partner with the share observed among
    # childless women of her age (FAM100N).
    with_partner_at_start = random.random(number_of_women) < (
        women_2010.share_of_childless_women_living_with_partner.reindex(woman_age_at_start).to_numpy())
    # Otherwise she moves in at a later age, or never. The share of ALL women
    # living with a partner at each age (FAM100N) is read as "the share who have
    # moved in by that age", and her age is drawn from the part of that curve
    # that lies beyond her age in 2010. The couple then stays together.
    share_moved_in_by_age = women_2010.share_of_all_women_living_with_partner.cummax()
    share_at_start = share_moved_in_by_age.reindex(woman_age_at_start).to_numpy()
    draw = share_at_start + random.random(number_of_women) * (1 - share_at_start)
    place_on_curve = np.searchsorted(share_moved_in_by_age.to_numpy(), draw)
    never_moves_in = place_on_curve >= len(share_moved_in_by_age)
    age_moving_in = share_moved_in_by_age.index.to_numpy()[np.minimum(place_on_curve,
                                                                     len(share_moved_in_by_age) - 1)]
    # She counts as living with him at the END of the year before that birthday-age
    # is reached on 1 January.
    year_moving_in = np.where(never_moves_in, 9999, first_year - 1 + age_moving_in - woman_age_at_start)
    year_moving_in[with_partner_at_start] = first_year - 1

    return {"person_id": 1_000_000 + np.arange(number_of_women),      # women's IDs start at 1,000,000
            "birth_year": woman_birth_year, "foreign_born": woman_foreign_born,
            "education_position": woman_education_position, "year_moving_in": year_moving_in}


def yearly_traits(random, women):
    """PART 2 - Follow the women through the years. Each year: positions carry
    over or are redrawn, partners move in, that year's employment, enrolment,
    education and wage are set, childless couples may conceive, and the
    register rows are written.

    Her partner, once she has one, lives in the man_* arrays at the same
    position as she does. Returns the rows of the five registers.
    """
    rate = load_yearly_rates()
    immigrant_share = (pd.read_csv(DST_DATA_FOLDER / "immigrant_share_2010.csv")
                       .set_index(["sex", "age"])["share_immigrant"])
    first_year = min(YEARS)
    woman_id, woman_birth_year = women["person_id"], women["birth_year"]
    woman_foreign_born, year_moving_in = women["foreign_born"], women["year_moving_in"]
    woman_education_position = women["education_position"]
    number_of_women = len(woman_id)
    # IDs: men from 5,000,000, children from 9,000,000.
    next_man_id, next_child_id = 5_000_000, 9_000_000

    # ---- the woman: what changes over the years -------------------------------------
    woman_position = new_positions(random, number_of_women)
    woman_high_education = np.zeros(number_of_women, bool)
    woman_had_first_conception = np.zeros(number_of_women, bool)

    # ---- her partner (man_id 0 = none yet) ---------------------------------------------
    # A man exists in the registers only from the year he moves in. His own
    # fixed traits (birth year, origin, education position) are drawn in that year.
    man_id = np.zeros(number_of_women, int)
    man_birth_year = np.zeros(number_of_women, int)
    man_foreign_born = np.zeros(number_of_women, bool)
    man_education_position = np.zeros(number_of_women)
    man_high_education = np.zeros(number_of_women, bool)
    man_position = new_positions(random, number_of_women)

    register_rows = {name: [] for name in REGISTER_FILES}        # one chunk of rows per year

    def add_register_rows(year, sex, person_id, birth_year, foreign_born, partner_id, employed, wage,
                          enrolled, high_education, education_unknown):
        """Append one year's rows for these people to the four person-year registers."""
        person_year = {"person_id": person_id, "year": year}
        # bef, population: origin is fixed for life; partner_id is the
        # co-resident partner at year end, 0 = no partner.
        register_rows["bef"].append(pd.DataFrame({
            **person_year, "sex": sex, "birth_year": birth_year,
            "origin": np.where(foreign_born, "foreign", "DK"), "partner_id": partner_id}))
        # ras, employment: 1 = employed, 0 = not employed.
        register_rows["ras"].append(pd.DataFrame({**person_year, "employed": employed.astype(int)}))
        # ind, income: annual labour income in DKK.
        register_rows["ind"].append(pd.DataFrame({**person_year, "wage_dkk": wage}))
        # udda, education: enrolled = currently in education; isced_level =
        # highest completed level, 3 = low (ISCED 0-4), 6 = high (ISCED 5-6),
        # empty = unknown.
        isced_level = np.where(high_education, 6, 3).astype(float)
        isced_level[education_unknown] = np.nan
        register_rows["udda"].append(pd.DataFrame({
            **person_year, "enrolled": enrolled.astype(int), "isced_level": isced_level}))

    # All registers describe the situation at the END of the year.
    for year in YEARS:
        woman_age = year - woman_birth_year

        # ---- positions carry over from last year, or are drawn afresh ---------------
        if year > first_year:
            for position in (woman_position, man_position):
                fresh = new_positions(random, number_of_women)
                for name in position:
                    keeps_position = random.random(number_of_women) < P_KEEP_LAST_YEARS_POSITION
                    position[name] = np.where(keeps_position, position[name], fresh[name])

        # ---- partners moving in this year ----------------------------------------------
        moving_in = np.nonzero((year_moving_in <= year) & (man_id == 0))[0]
        man_id[moving_in] = next_man_id + np.arange(len(moving_in))
        next_man_id += len(moving_in)
        # His age = her age + a normally distributed gap, kept within 18-65. He
        # needs an age because his employment and wage are drawn from the
        # figures for his sex and age.
        man_age_now = np.clip(np.round(woman_age[moving_in] + random.normal(
            *PARTNER_AGE_GAP_MEAN_AND_SD, len(moving_in))), 18, 65).astype(int)
        man_birth_year[moving_in] = year - man_age_now
        # Born abroad: the share of immigrants among men of his age in 2010 (FOLK1E).
        man_age_at_start = np.clip(first_year - 1 - man_birth_year[moving_in],
                                   immigrant_share["M"].index.min(), immigrant_share["M"].index.max())
        man_foreign_born[moving_in] = random.random(len(moving_in)) < immigrant_share["M"].reindex(
            man_age_at_start).to_numpy()
        man_education_position[moving_in] = random.random(len(moving_in))

        # ---- this year's traits ------------------------------------------------------------
        man_age = year - man_birth_year
        woman_employed, woman_enrolled, woman_education_unknown, woman_wage = traits_this_year(
            rate, "F", year, woman_age, woman_position, woman_education_position, woman_high_education)
        man_employed, man_enrolled, man_education_unknown, man_wage = traits_this_year(
            rate, "M", year, man_age, man_position, man_education_position, man_high_education)

        # ---- conception ----------------------------------------------------------------------
        # At risk: living with a partner, no first conception yet, aged 18-45.
        # The probability comes from the logit equation above, evaluated on this
        # year's traits. A success is a conception now and a birth next year, on
        # a random day; the father is her partner.
        at_risk = np.nonzero((man_id > 0) & ~woman_had_first_conception
                             & (woman_age >= 18) & (woman_age <= 45))[0]
        earner_constellation = constellation(woman_employed[at_risk], man_employed[at_risk],
                                             woman_wage[at_risk], man_wage[at_risk])
        # Household-wage terciles among the couples at risk THIS year.
        household_wage_tercile = wage_tercile(woman_wage[at_risk] + man_wage[at_risk])
        # The same education categories that stage 2 will derive from the registers.
        education = np.select(
            [woman_enrolled[at_risk], woman_education_unknown[at_risk], woman_high_education[at_risk]],
            ["enrolled", "unknown", "high"], "low")
        p_conceive = birth_probability(
            woman_age_group=age_group(woman_age[at_risk]),
            earner_constellation=earner_constellation,
            household_wage_tercile=household_wage_tercile,
            woman_education=education,
            either_partner_foreign=(woman_foreign_born[at_risk] | man_foreign_born[at_risk]).astype(int))
        conceiving = at_risk[random.random(len(at_risk)) < p_conceive]
        woman_had_first_conception[conceiving] = True
        day_of_next_year = random.integers(0, 365, len(conceiving))
        register_rows["fod"].append(pd.DataFrame({
            "child_id": next_child_id + np.arange(len(conceiving)),
            "mother_id": woman_id[conceiving], "father_id": man_id[conceiving],
            "birth_date": pd.to_datetime(f"{year + 1}-01-01") + pd.to_timedelta(day_of_next_year, "D")}))
        next_child_id += len(conceiving)

        # ---- write this year's register rows ----------------------------------------------------
        # One row per woman aged 18 or over, and one per partner.
        women = np.nonzero(woman_age >= 18)[0]
        add_register_rows(year, "F", woman_id[women], woman_birth_year[women], woman_foreign_born[women],
                          man_id[women], woman_employed[women], woman_wage[women], woman_enrolled[women],
                          woman_high_education[women], woman_education_unknown[women])
        partnered = women[man_id[women] > 0]
        add_register_rows(year, "M", man_id[partnered], man_birth_year[partnered],
                          man_foreign_born[partnered], woman_id[partnered], man_employed[partnered],
                          man_wage[partnered], man_enrolled[partnered], man_high_education[partnered],
                          man_education_unknown[partnered])

    return register_rows


def simulate():
    """Run the simulation and write the five register files."""
    random = np.random.default_rng(SEED)
    women = fixed_traits(random)
    register_rows = yearly_traits(random, women)

    SIMULATED_REGISTER_FOLDER.mkdir(exist_ok=True)
    for name, yearly_chunks in register_rows.items():
        register = pd.concat(yearly_chunks, ignore_index=True)
        register.to_csv(REGISTER_FILES[name], index=False)
        print(f"  {REGISTER_FILES[name].name}: {len(register):,} rows")


if __name__ == "__main__":
    print("Stage 1: simulating raw registers ...")
    simulate()
