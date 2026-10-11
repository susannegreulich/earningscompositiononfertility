"""
STAGE 1 - Simulate raw, person-level registers (the data that never leaves
Statistics Denmark in the real study).

Every behavioural assumption is a parameter in the block below. Edit a value,
rerun `python run_all.py`, and the couple-year file, the aggregate table and
the model estimates all change accordingly.

Women born 1965-2002 are followed year by year from age 18 to 46. Each year,
in this order:
  1. a conception from last year becomes a birth (first births only)
  2. education
  3. unions break up or form (a new partner is drawn)
  4. jobs are found or lost, for her and her partner
  5. wages for both
  6. childless, partnered women aged 18-45 may conceive
"""
import numpy as np
import pandas as pd

from common import (AGE_GROUPS, CONSTELLATIONS, OUTPUT_FOLDER, REGISTER_FILES, age_group,
                    constellation)

# =============================================================================
# PARAMETERS - edit these. Names starting with P_ are yearly probabilities.
# =============================================================================
SEED = 2026
WOMEN_PER_COHORT = 4_000            # Denmark has ~33,000; 4,000 runs in ~30 s
BIRTH_COHORTS = range(1965, 2003)   # birth years of the simulated women
REGISTER_YEARS = range(2009, 2021)  # years written to the register files

# ---- FIRST BIRTH ------------------------------------------------------------
# P(conception leading to a first birth within the year) for a couple in the
# REFERENCE group on the other traits: woman low-educated, household wages in
# the low tercile, both partners born in Denmark.
# Rows: woman's age group. Columns: employment-wage constellation.
# Source: the paper's Danish Model 2 (Table A2), converted to probabilities.
FIRST_BIRTH_PROBABILITY = pd.DataFrame(
    # dual     both     male     female   dual male  dual female
    # equal    jobless  works    works    main       main
    # earners                             earner     earner
    [[0.0717,  0.0554,  0.0644,  0.0604,  0.0578,    0.0730],   # 18-25
     [0.0928,  0.0720,  0.0835,  0.0784,  0.0751,    0.0945],   # 26-30
     [0.0861,  0.0667,  0.0774,  0.0727,  0.0696,    0.0876],   # 31-35
     [0.0213,  0.0162,  0.0190,  0.0177,  0.0169,    0.0217]],  # 36-45
    index=AGE_GROUPS, columns=CONSTELLATIONS)

# Other traits shift the ODDS multiplicatively (odds ratios, 1 = no effect).
# Source: the paper's Danish Model 2.
FIRST_BIRTH_ODDS_RATIO_EDUCATION = {"low": 1.0, "enrolled": 0.797, "high": 1.366, "unknown": 0.998}
FIRST_BIRTH_ODDS_RATIO_HOUSEHOLD_WAGE = {"low": 1.0, "middle": 1.723, "high": 2.534}  # yearly terciles
FIRST_BIRTH_ODDS_RATIO_FOREIGN = 0.848                                                # either partner

# ---- SOURCES ----------------------------------------------------------------
# Everything below the first-birth block is set from Statistics Denmark's
# StatBank (statistikbanken.dk), averaged over 2010-2019, the paper's
# observation period (LONS50 starts in 2013 and LONS11 in 2015). Each comment
# names the table. "Set so that ..." marks a transition probability chosen by
# hand to reproduce a published level, because StatBank publishes stocks (who
# is employed, enrolled, in a couple), not yearly flows. "Assumption" marks a
# value with no StatBank source.

# ---- EDUCATION (woman) --------------------------------------------------------
P_ENROLLED_AT_18 = 0.80           # UDDAKT10 students / FOLK1E women aged 18
# Yearly P(leave education) and P(enrol | low-educated), by age from key on.
# Set so that the share enrolled by age follows UDDAKT10 / FOLK1E:
# 52% at 19, 45% at 20, 62% at 22, 42% at 25, 20% at 28, 13% at 30, 7% at 35.
P_FINISH_STUDIES = {18: 0.62, 21: 0.27, 25: 0.22, 30: 0.13}
P_ENROL_IF_LOW = {18: 0.33, 21: 0.41, 25: 0.0}
P_HIGH_IF_FINISH_BY_21 = 0.05     # assumption: leaving by 21 is upper secondary
# Set so that about 54% of women aged 30-34 hold a higher degree (HFUDD11: 53.6%).
P_HIGH_IF_FINISH_AFTER_21 = 0.54
# Share with education "not stated", women 25-44 (HFUDD11).
P_EDUCATION_UNKNOWN = {"native": 0.003, "foreign": 0.078}

# ---- ORIGIN -----------------------------------------------------------------
P_WOMAN_FOREIGN = 0.15            # immigrants among women 18-45, FOLK1E
# Assumption: 60% of foreign-born women have a foreign-born partner. The value
# for native women then makes 14% of men foreign-born (FOLK1E, men 18-45).
P_MAN_FOREIGN = {"woman_native": 0.06, "woman_foreign": 0.60}

# ---- UNIONS -----------------------------------------------------------------
P_PARTNERED_AT_18 = 0.04          # women aged 18 in a couple, FAM100N / FOLK1E
# Set so that the share of women living in a CHILDLESS couple follows FAM100N /
# FOLK1E (17% at 20, 33% at 25, 21% at 30, 9% at 35, 7% at 40). Childless
# couples are the population the couple-year sample is drawn from, so this
# gives the sample the paper's age mix (Table A1). The simulation has fewer
# mothers than Denmark, so the share of ALL women in a couple comes out too low
# after 30 (52% at 40, against 74%).
P_UNION_FORM = {"18-25": 0.14, "26-30": 0.18, "31-35": 0.055, "36-45": 0.05}
# Assumption: a woman without a job is half as likely to form a union.
# Together with P_MAN_EMPLOYED_AT_UNION this makes partnered people more often
# employed than the population, as in the paper's sample (women 74%, men 80%).
UNION_FORMATION_MULTIPLIER_IF_NOT_EMPLOYED = 0.5
# parent: children living with both parents who no longer do a year later
# (BRN11: 3.3%). childless: assumption, no StatBank source.
P_UNION_DISSOLVE = {"childless": 0.14, "parent": 0.033}
# (his age - her age). Mean: first-time fathers minus first-time mothers
# (FOD11). Sd: age difference at marriage, brides 18-46 (VIE825).
PARTNER_AGE_GAP_MEAN_AND_SD = (2.2, 4.9)
# Assumption on the gradient; the level gives 39% of men aged 30-34 a higher
# degree (HFUDD11).
P_MAN_HIGH_EDUCATION = {"woman_low": 0.22, "woman_enrolled": 0.40, "woman_high": 0.55}
P_MAN_STUDENT_IF_UNDER_25 = 0.54  # men 18-24 enrolled, UDDAKT10 / FOLK1E
# Assumption: 10 points above the employment rate of all men of that age in
# RAS200 (62/73/81/85%), because men who move in with a partner are more often
# employed.
P_MAN_EMPLOYED_AT_UNION = {"18-25": 0.72, "26-30": 0.83, "31-35": 0.91, "36-45": 0.93}

# ---- EMPLOYMENT (yearly transitions) -------------------------------------------
# P(get a job | not employed last year) and P(lose job | employed last year),
# for the reference person: low education, not a student, born in Denmark,
# partner employed. Set so that the simulated employment rate of people
# without higher education matches RAS209 at ages 20-24 / 25-29 / 30-34 /
# 35-44 (women 63/61/65/73%, men 62/71/78/81%). The reference person's rate,
# find / (find + lose), is higher than that because the multipliers below pull
# the average down. Assumption: find + lose, the yearly turnover, which RAS
# does not publish.
P_FIND_JOB = {"F": {"18-25": 0.38, "26-30": 0.34, "31-35": 0.38, "36-45": 0.40},
              "M": {"18-25": 0.53, "26-30": 0.52, "31-35": 0.49, "36-45": 0.44}}
P_LOSE_JOB = {"F": {"18-25": 0.12, "26-30": 0.13, "31-35": 0.13, "36-45": 0.10},
              "M": {"18-25": 0.12, "26-30": 0.11, "31-35": 0.09, "36-45": 0.06}}
P_EMPLOYED_AT_18 = 0.55           # employment rate, women 18-19, RAS200
# Multipliers on those probabilities. For education and origin the odds ratio
# of employment is split evenly: find x sqrt(OR), lose / sqrt(OR).
# Higher vs. no higher education, ages 25-44, RAS209: odds ratio 2.1.
EMPLOYMENT_MULTIPLIER_HIGH_EDUCATION = {"find": 1.46, "lose": 0.68}
# Immigrants vs. Danish origin, ages 25-44, RAS200: odds ratio 0.38.
EMPLOYMENT_MULTIPLIER_FOREIGN = {"find": 0.61, "lose": 1.63}
# Assumptions: students, and people whose partner had no job (shared shocks).
EMPLOYMENT_MULTIPLIER_STUDENT = {"find": 0.85, "lose": 1.20}
EMPLOYMENT_MULTIPLIER_PARTNER_NOT_EMPLOYED = {"find": 0.55, "lose": 1.60}

# ---- WAGES (annual labour income, DKK) ---------------------------------------------
# Full-year wage at age 30-34 by sex and education.
# Men's level: set so that the simulated average wage of men aged 30-34 with
# wage income matches INDKP111 (353,138).
# Women's level: 90% of men's, the female / male ratio of standardised
# (full-time) monthly earnings at 30-34 in LONS50. INDKP111 gives 80%, but its
# women include mothers on leave and in part-time work, who are not in the
# childless couple-year sample.
# High / low ratio: earnings premium of higher education in LONS11
# (women x1.32, men x1.43).
WAGE_BASE = {"F": {"low": 299_000, "high": 395_000},
             "M": {"low": 312_000, "high": 446_000}}
# Age profile, multiplier relative to 30-34, interpolated: standardised monthly
# earnings by age group, LONS50 (under 20, 20-24, ..., 45-49).
WAGE_AGE_MULTIPLIER = {"F": {18: 0.48, 22: 0.65, 27: 0.88, 32: 1.00, 37: 1.08, 42: 1.13, 47: 1.15},
                       "M": {18: 0.43, 22: 0.65, 27: 0.86, 32: 1.00, 37: 1.12, 42: 1.20, 47: 1.24}}
# The remaining wage parameters are assumptions: StatBank has no panel data on
# individual wage dynamics or on couples' wages. They are chosen so that dual
# earners split into equal and unequal earners roughly as in the paper's Table
# A1 (47% equal, 38% male main earner, 16% female main earner). Less wage
# spread (person effect, shock, spells) means more equal earners.
WAGE_PERSON_EFFECT_LOG_SD = 0.25               # log sd of a permanent person effect
WAGE_PERSON_EFFECT_COUPLE_CORRELATION = 0.30   # correlation of partners' person effects
WAGE_SHOCK_LOG_SD = 0.25                       # log sd of the persistent yearly shock
WAGE_SHOCK_PERSISTENCE = 0.5                   # share of last year's shock that carries over
P_LOW_EARNINGS_SPELL_START = 0.04              # a part-year / low-hours year begins
P_LOW_EARNINGS_SPELL_END = 0.5                 # ... and ends
LOW_EARNINGS_SPELL_FACTOR = 0.50               # wage x this during such a spell
STUDENT_JOB_FACTOR = 0.70                      # employed students earn this share
P_INCOME_IF_NOT_EMPLOYED = 0.40                # some labour income despite not employed
INCOME_IF_NOT_EMPLOYED_MEDIAN = 30_000
INCOME_IF_NOT_EMPLOYED_LOG_SD = 0.7
# =============================================================================


def value_for_age(table_by_starting_age, age):
    """Look up a table keyed by the age from which a value applies, e.g. {18: .., 21: ..}."""
    starting_ages = np.array(sorted(table_by_starting_age))
    values = np.array([table_by_starting_age[a] for a in starting_ages])
    return values[np.searchsorted(starting_ages, age, side="right") - 1]


def value_for_age_group(table_by_age_group, age):
    """Look up a table keyed by the paper's age groups, e.g. {"18-25": .., "26-30": ..}."""
    return np.array([table_by_age_group[group] for group in age_group(age)])


def wage_age_multiplier(sex, age):
    """Wage relative to age 30-34, interpolated between the ages in WAGE_AGE_MULTIPLIER."""
    profile = WAGE_AGE_MULTIPLIER[sex]
    return np.interp(age, list(profile), list(profile.values()))


def birth_probability(woman_age_group, earner_constellation, woman_education,
                      household_wage_tercile, either_partner_foreign):
    """The yearly probability of a conception leading to a first birth."""
    # Start from the table for the reference group (age group x constellation)...
    probability = FIRST_BIRTH_PROBABILITY.to_numpy()[
        [AGE_GROUPS.index(group) for group in woman_age_group],
        [CONSTELLATIONS.index(c) for c in earner_constellation]]
    # ...then multiply the odds by the odds ratio of each other trait.
    odds = probability / (1 - probability)
    odds *= np.array([FIRST_BIRTH_ODDS_RATIO_EDUCATION[e] for e in woman_education])
    odds *= np.array([FIRST_BIRTH_ODDS_RATIO_HOUSEHOLD_WAGE[t] for t in household_wage_tercile])
    odds *= np.where(np.asarray(either_partner_foreign, bool), FIRST_BIRTH_ODDS_RATIO_FOREIGN, 1.0)
    return odds / (1 + odds)


def simulate():
    """Run the yearly simulation and write the five register files.

    Every woman is one position in a set of numpy arrays. Her current partner,
    if any, lives in the parallel man_* arrays at the same position, so "woman
    i and her partner" is always position i. Each calendar year updates all
    women at once.
    """
    random = np.random.default_rng(SEED)
    woman_birth_year = np.repeat(np.array(BIRTH_COHORTS), WOMEN_PER_COHORT)
    number_of_women = len(woman_birth_year)
    # Person IDs: women from 1,000,000, men from 5,000,000, children from 9,000,000.
    woman_id = 1_000_000 + np.arange(number_of_women)
    next_man_id, next_child_id = 5_000_000, 9_000_000

    # ---- the woman: traits fixed for life -----------------------------------------
    woman_foreign_born = random.random(number_of_women) < P_WOMAN_FOREIGN
    woman_education_unknown = random.random(number_of_women) < np.where(
        woman_foreign_born, P_EDUCATION_UNKNOWN["foreign"], P_EDUCATION_UNKNOWN["native"])
    woman_wage_person_effect = random.normal(0, WAGE_PERSON_EFFECT_LOG_SD, number_of_women)
    # ---- the woman: state updated every year ----------------------------------------
    woman_education = np.full(number_of_women, "low", dtype=object)   # enrolled / low / high
    woman_employed = np.zeros(number_of_women, bool)
    woman_wage_shock = np.zeros(number_of_women)                 # persistent yearly log-wage shock
    woman_in_low_earnings_spell = np.zeros(number_of_women, bool)
    woman_has_child = np.zeros(number_of_women, bool)
    woman_pregnant = np.zeros(number_of_women, bool)             # conceived this year, birth next year
    # ---- her current partner (man_id 0 = no partner) ----------------------------------
    # A man exists only while he is the partner of a simulated woman. If the
    # union ends he disappears; a later partner of the same woman gets a new ID.
    man_id = np.zeros(number_of_women, int)
    man_birth_year = np.zeros(number_of_women, int)
    man_foreign_born = np.zeros(number_of_women, bool)
    man_high_education = np.zeros(number_of_women, bool)         # fixed for the union
    man_student = np.zeros(number_of_women, bool)                # ends at age 25
    man_employed = np.zeros(number_of_women, bool)
    man_wage_person_effect = np.zeros(number_of_women)
    man_wage_shock = np.zeros(number_of_women)
    man_in_low_earnings_spell = np.zeros(number_of_women, bool)

    register_rows = {name: [] for name in REGISTER_FILES}        # one chunk of rows per year

    def draw_new_partners(women, year):
        """Give each of these women a brand-new partner (union formation)."""
        nonlocal next_man_id
        count = len(women)
        man_id[women] = next_man_id + np.arange(count)
        next_man_id += count
        # His age = her age + a normally distributed gap, kept within 18-65.
        woman_age = year - woman_birth_year[women]
        man_age = np.clip(np.round(woman_age + random.normal(*PARTNER_AGE_GAP_MEAN_AND_SD, count)),
                          18, 65).astype(int)
        man_birth_year[women] = year - man_age
        # Origin homogamy: he is more likely foreign-born if she is.
        man_foreign_born[women] = random.random(count) < np.where(
            woman_foreign_born[women], P_MAN_FOREIGN["woman_foreign"], P_MAN_FOREIGN["woman_native"])
        # Assortative education: P(he is high-educated) depends on her current status.
        man_high_education[women] = random.random(count) < np.array(
            [P_MAN_HIGH_EDUCATION["woman_" + status] for status in woman_education[women]])
        man_student[women] = (man_age < 25) & (random.random(count) < P_MAN_STUDENT_IF_UNDER_25)
        man_employed[women] = random.random(count) < value_for_age_group(P_MAN_EMPLOYED_AT_UNION, man_age)
        # His permanent wage effect is correlated with hers (assortative earnings).
        correlation = WAGE_PERSON_EFFECT_COUPLE_CORRELATION
        man_wage_person_effect[women] = (
            correlation * woman_wage_person_effect[women]
            + np.sqrt(1 - correlation ** 2) * random.normal(0, WAGE_PERSON_EFFECT_LOG_SD, count))
        man_wage_shock[women] = random.normal(0, WAGE_SHOCK_LOG_SD, count)
        man_in_low_earnings_spell[women] = False

    def employment_this_year(employed_last_year, sex, age, student, high_education, foreign_born,
                             partner_employed_last_year):
        """One year of job transitions.

        Someone employed last year loses the job with P_LOSE_JOB; someone not
        employed finds one with P_FIND_JOB. So this year's status depends on
        last year's status, but on nothing further back.
        """
        p_find = value_for_age_group(P_FIND_JOB[sex], age)
        p_lose = value_for_age_group(P_LOSE_JOB[sex], age)
        # Both probabilities are multiplied by every multiplier that applies.
        for multiplier, applies in (
                (EMPLOYMENT_MULTIPLIER_STUDENT, student),
                (EMPLOYMENT_MULTIPLIER_HIGH_EDUCATION, high_education),
                (EMPLOYMENT_MULTIPLIER_FOREIGN, foreign_born),
                (EMPLOYMENT_MULTIPLIER_PARTNER_NOT_EMPLOYED, ~partner_employed_last_year)):
            p_find = np.where(applies, p_find * multiplier["find"], p_find)
            p_lose = np.where(applies, p_lose * multiplier["lose"], p_lose)
        draw = random.random(len(employed_last_year))
        return np.where(employed_last_year, draw >= np.clip(p_lose, 0, 1), draw < np.clip(p_find, 0, 1))

    def annual_wage(employed, sex, age, high_education, student, person_effect, wage_shock,
                    in_low_earnings_spell):
        """Annual labour income in DKK, rounded to 100."""
        # Employed: base wage (sex, education) x age profile x exp(person effect + shock)...
        base_wage = np.where(high_education, WAGE_BASE[sex]["high"], WAGE_BASE[sex]["low"])
        wage = base_wage * wage_age_multiplier(sex, age) * np.exp(person_effect + wage_shock)
        # ...reduced in a part-year / low-hours year, and for student jobs.
        wage = np.where(in_low_earnings_spell, wage * LOW_EARNINGS_SPELL_FACTOR, wage)
        wage = np.where(student, wage * STUDENT_JOB_FACTOR, wage)
        # Not employed: usually 0, but sometimes a small income from part-year
        # work. This is why the income register can show wages for someone the
        # employment register counts as not employed.
        count = len(employed)
        income_if_not_employed = np.where(
            random.random(count) < P_INCOME_IF_NOT_EMPLOYED,
            INCOME_IF_NOT_EMPLOYED_MEDIAN * np.exp(random.normal(0, INCOME_IF_NOT_EMPLOYED_LOG_SD, count)),
            0.0)
        return np.round(np.where(employed, wage, income_if_not_employed), -2)

    def advance_wage_shocks(wage_shock, in_low_earnings_spell):
        """Move the persistent wage shock and the low-earnings spell one year on."""
        count = len(wage_shock)
        # The shock keeps a share WAGE_SHOCK_PERSISTENCE of last year's value (an AR(1) process).
        wage_shock = (WAGE_SHOCK_PERSISTENCE * wage_shock
                      + np.sqrt(1 - WAGE_SHOCK_PERSISTENCE ** 2) * random.normal(0, WAGE_SHOCK_LOG_SD, count))
        # A spell continues unless it ends; outside a spell, one may start.
        in_low_earnings_spell = np.where(in_low_earnings_spell,
                                         random.random(count) >= P_LOW_EARNINGS_SPELL_END,
                                         random.random(count) < P_LOW_EARNINGS_SPELL_START)
        return wage_shock, in_low_earnings_spell

    def add_register_rows(year, sex, person_id, birth_year, foreign_born, partner_id, employed, wage,
                          enrolled, isced_level):
        """Append one year's rows for these people to the four person-year registers."""
        person_year = {"person_id": person_id, "year": year}
        # bef, population: origin is fixed for life; partner_id is the
        # co-resident partner at year end, 0 = no partner.
        register_rows["bef"].append(pd.DataFrame({
            **person_year, "sex": sex, "birth_year": birth_year,
            "origin": np.where(foreign_born, "foreign", "DK"), "partner_id": partner_id}))
        # ras, employment: 1 = employed or self-employed, 0 = not employed
        # (student, unemployed, inactive).
        register_rows["ras"].append(pd.DataFrame({**person_year, "employed": employed.astype(int)}))
        # ind, income: annual labour income in DKK.
        register_rows["ind"].append(pd.DataFrame({**person_year, "wage_dkk": wage}))
        # udda, education: enrolled = currently in education; isced_level =
        # highest completed level, 3 = low (ISCED 0-4), 6 = high (ISCED 5-6),
        # empty = unknown.
        register_rows["udda"].append(pd.DataFrame({
            **person_year, "enrolled": enrolled.astype(int), "isced_level": isced_level}))

    # One pass per calendar year, from the year the oldest cohort turns 18 to
    # the last register year. All registers describe the situation at the END
    # of the year, after that year's steps.
    for year in range(min(BIRTH_COHORTS) + 18, max(REGISTER_YEARS) + 1):
        woman_age = year - woman_birth_year
        in_simulation = (woman_age >= 18) & (woman_age <= 46)

        # ---- entry at 18: initial education, job and (rarely) a partner -----------
        turning_18 = np.nonzero(woman_age == 18)[0]
        woman_education[turning_18] = np.where(
            random.random(len(turning_18)) < P_ENROLLED_AT_18, "enrolled", "low")
        woman_employed[turning_18] = random.random(len(turning_18)) < P_EMPLOYED_AT_18
        partnered_at_18 = turning_18[random.random(len(turning_18)) < P_PARTNERED_AT_18]
        draw_new_partners(partnered_at_18, year)

        # Steps 1-4 apply to the women who were already in the simulation last year.
        continuing = np.nonzero(in_simulation & (woman_age > 18))[0]
        age_of_continuing = woman_age[continuing]

        # ---- 1. conceptions from last year become births -----------------------------
        # The birth date is a random day this year; the father is her partner
        # at conception (unions cannot end during a pregnancy, see step 3).
        giving_birth = continuing[woman_pregnant[continuing]]
        if len(giving_birth):
            day_of_year = random.integers(0, 365, len(giving_birth))
            register_rows["fod"].append(pd.DataFrame({
                "child_id": next_child_id + np.arange(len(giving_birth)),
                "mother_id": woman_id[giving_birth], "father_id": man_id[giving_birth],
                "birth_date": pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(day_of_year, "D")}))
            next_child_id += len(giving_birth)
            woman_has_child[giving_birth] = True
            woman_pregnant[giving_birth] = False

        # ---- 2. education ------------------------------------------------------------------
        education = woman_education[continuing]
        # Enrolled women leave education with a probability that depends on age...
        leaves_education = (education == "enrolled") & (
            random.random(len(continuing)) < value_for_age(P_FINISH_STUDIES, age_of_continuing))
        # ...with a higher degree if they leave after 21, mostly without one before.
        gets_high_degree = random.random(len(continuing)) < np.where(
            age_of_continuing <= 21, P_HIGH_IF_FINISH_BY_21, P_HIGH_IF_FINISH_AFTER_21)
        education = np.where(leaves_education, np.where(gets_high_degree, "high", "low"), education)
        # Low-educated women may enrol again (gap years).
        enrols_again = (education == "low") & (
            random.random(len(continuing)) < value_for_age(P_ENROL_IF_LOW, age_of_continuing))
        woman_education[continuing] = np.where(enrols_again, "enrolled", education)
        # Partners stop being students at 25.
        man_age = year - man_birth_year
        man_student[continuing] = man_student[continuing] & (man_age[continuing] < 25)

        # ---- 3. unions -------------------------------------------------------------------------
        # Existing unions break up, at different rates for childless couples
        # and parents, and never during a pregnancy.
        has_partner = man_id[continuing] > 0
        p_break_up = np.where(woman_has_child[continuing],
                              P_UNION_DISSOLVE["parent"], P_UNION_DISSOLVE["childless"])
        breaking_up = continuing[has_partner & (random.random(len(continuing)) < p_break_up)
                                 & ~woman_pregnant[continuing]]
        man_id[breaking_up] = 0
        # Singles, including those who just split, may form a union with a
        single = continuing[man_id[continuing] == 0]
        # newly drawn partner. Women without a job (last year's status) are less
        # likely to do so: selection into unions by employment.
        p_form_union = value_for_age_group(P_UNION_FORM, woman_age[single]) * np.where(
            woman_employed[single], 1.0, UNION_FORMATION_MULTIPLIER_IF_NOT_EMPLOYED)
        forming_union = single[random.random(len(single)) < p_form_union]
        draw_new_partners(forming_union, year)

        # ---- 4. employment -------------------------------------------------------------------------
        # Women and the partners they already had last year make job
        # transitions based on LAST year's status. A partner who joined this
        # year keeps the status drawn in draw_new_partners(). A woman without a
        # partner is treated like one whose partner is employed.
        with_last_years_partner = np.setdiff1d(continuing[man_id[continuing] > 0], forming_union)
        woman_employed_last_year = woman_employed.copy()
        woman_employed[continuing] = employment_this_year(
            woman_employed[continuing], "F", age_of_continuing,
            woman_education[continuing] == "enrolled", woman_education[continuing] == "high",
            woman_foreign_born[continuing],
            np.where(man_id[continuing] > 0, man_employed[continuing], True))
        man_employed[with_last_years_partner] = employment_this_year(
            man_employed[with_last_years_partner], "M", man_age[with_last_years_partner],
            man_student[with_last_years_partner], man_high_education[with_last_years_partner],
            man_foreign_born[with_last_years_partner],
            woman_employed_last_year[with_last_years_partner])
        woman_wage_shock[continuing], woman_in_low_earnings_spell[continuing] = advance_wage_shocks(
            woman_wage_shock[continuing], woman_in_low_earnings_spell[continuing])
        (man_wage_shock[with_last_years_partner],
         man_in_low_earnings_spell[with_last_years_partner]) = advance_wage_shocks(
            man_wage_shock[with_last_years_partner], man_in_low_earnings_spell[with_last_years_partner])

        # ---- 5. wages, for every woman in the simulation and every current partner ------------------
        women = np.nonzero(in_simulation)[0]
        woman_wage = np.zeros(number_of_women)
        man_wage = np.zeros(number_of_women)
        woman_wage[women] = annual_wage(
            woman_employed[women], "F", woman_age[women], woman_education[women] == "high",
            woman_education[women] == "enrolled", woman_wage_person_effect[women],
            woman_wage_shock[women], woman_in_low_earnings_spell[women])
        partnered = women[man_id[women] > 0]
        man_wage[partnered] = annual_wage(
            man_employed[partnered], "M", man_age[partnered], man_high_education[partnered],
            man_student[partnered], man_wage_person_effect[partnered],
            man_wage_shock[partnered], man_in_low_earnings_spell[partnered])

        # ---- 6. conception ---------------------------------------------------------------------------
        # At risk: partnered, childless, not already pregnant, aged 18-45.
        # The probability is re-evaluated EVERY year from that year's traits,
        # so a couple that ages into the next group, loses a job or gets a pay
        # rise faces a new risk. A success is a conception now and a birth next
        # year. No conceptions in the last register year, because the birth
        # would fall outside the registers.
        at_risk = women[(man_id[women] > 0) & ~woman_has_child[women] & ~woman_pregnant[women]
                        & (woman_age[women] <= 45)]
        if len(at_risk) and year < max(REGISTER_YEARS):
            earner_constellation = constellation(woman_employed[at_risk], man_employed[at_risk],
                                                 woman_wage[at_risk], man_wage[at_risk])
            # Household-wage terciles among the couples at risk THIS year.
            household_wage = woman_wage[at_risk] + man_wage[at_risk]
            tercile_limits = np.quantile(household_wage, [1 / 3, 2 / 3])
            household_wage_tercile = np.array(["low", "middle", "high"])[
                np.digitize(household_wage, tercile_limits)]
            education = np.where(woman_education_unknown[at_risk], "unknown", woman_education[at_risk])
            p_conceive = birth_probability(
                age_group(woman_age[at_risk]), earner_constellation, education,
                household_wage_tercile, woman_foreign_born[at_risk] | man_foreign_born[at_risk])
            woman_pregnant[at_risk] = random.random(len(at_risk)) < p_conceive

        # ---- write this year's register rows --------------------------------------------------------------
        # One row per woman in the simulation (aged 18-46) and one per current
        # partner. Women with unknown education have an empty isced_level and
        # enrolled = 0 in every year; men's isced_level is never empty.
        if year in REGISTER_YEARS:
            isced_level = np.where(woman_education[women] == "high", 6, 3).astype(float)
            isced_level[woman_education_unknown[women]] = np.nan
            enrolled = (woman_education[women] == "enrolled") & ~woman_education_unknown[women]
            add_register_rows(year, "F", woman_id[women], woman_birth_year[women],
                              woman_foreign_born[women], man_id[women], woman_employed[women],
                              woman_wage[women], enrolled, isced_level)
            add_register_rows(year, "M", man_id[partnered], man_birth_year[partnered],
                              man_foreign_born[partnered], woman_id[partnered], man_employed[partnered],
                              man_wage[partnered], man_student[partnered],
                              np.where(man_high_education[partnered], 6, 3))

    # The birth register (fod) holds the FULL birth history, also births before
    # REGISTER_YEARS, as the real one does. Don't truncate it: stage 2 needs it
    # to know who is still childless. Writing only births from REGISTER_YEARS
    # made earlier mothers look childless, which diluted all effects.
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    for name, yearly_chunks in register_rows.items():
        register = pd.concat(yearly_chunks, ignore_index=True)
        register.to_csv(REGISTER_FILES[name], index=False)
        print(f"  {REGISTER_FILES[name].name}: {len(register):,} rows")


if __name__ == "__main__":
    print("Stage 1: simulating raw registers ...")
    simulate()
