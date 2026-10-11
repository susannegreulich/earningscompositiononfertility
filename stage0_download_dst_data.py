"""
STAGE 0 - Download every Statistics Denmark figure that stage 1 draws from, and
save them as three small tables in 0dst_data/, with a chart of each figure as a
visual check. Nothing here is simulated: these are published counts and rates.
The tables are kept in the repository, so this stage only needs to run again
to refresh them.

Source: StatBank (statistikbanken.dk) through its public API, no login needed.
Every download below is preceded by a comment naming the StatBank table, which
can be opened at statistikbanken.dk/<table name>.

It also holds the paper's Table A2 (Model 2), typed in, and turns it into a
table of Danish odds ratios: paper_table_a2_model2_denmark.csv. See the section
"THE PAPER'S TABLE A2" below for how the Danish numbers are computed.

Statistics Denmark tables used, and the file each ends up in:
  women_2010.csv              FOLK1E, BARNLOS, FAM100N
  immigrant_share_2010.csv    FOLK1E
  yearly_rates.csv            RAS200, HFUDD11, INDKP111, LONS50, UDDAKT10, FOLK1E
"""
import io
import json
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from common import DST_DATA_FOLDER

START_YEAR = 2010                       # traits that are drawn once are taken from this year
YEARS = range(2010, 2020)               # traits that are drawn yearly are taken from each of these
YOUNGEST_AGE, OLDEST_AGE = 18, 65       # ages covered by the yearly table
SEX_NAMES = {"K": "F", "2": "F", "M": "M", "1": "M"}     # StatBank's codes -> F / M

# How much older a man is than his partner: mean and standard deviation, in
# years. This is the one Statistics Denmark figure that is typed in rather than
# downloaded. Stage 1 uses it to give each partner an age.
# Mean 2.3: average age of first-time fathers (31.3) minus first-time mothers
#   (29.0) in 2010, TABLE FOD11 "Average age of women giving birth and new fathers".
# Sd 5.1: spread of (his age - her age) over all marriages in 2010 with a bride
#   aged 18-46, TABLE VIE825 "Marriages".
PARTNER_AGE_GAP_MEAN_AND_SD = (2.3, 5.1)


# =============================================================================
# THE PAPER'S TABLE A2, MODEL 2 - turned into a table for Denmark only
# =============================================================================
# Table A2 pools five countries and uses NORWAY as the reference country, so it
# never prints a Danish odds ratio. For each trait it prints two rows:
#
#   main row         e.g. "Jobless couple"            0.670
#       The odds ratio in Norway: the odds of conceiving for a jobless couple
#       relative to a dual equal-earner couple, both in Norway.
#   Denmark row      e.g. "Denmark * Jobless couple"  1.132
#       A ratio of two odds ratios: how many times larger the Danish odds ratio
#       is than the Norwegian one. It is NOT "Danish jobless couples compared
#       with Norwegian jobless couples".
#
# The Danish odds ratio is their product:
#
#   Danish odds ratio = main row x Denmark row = 0.670 x 1.132 = 0.758
#
#   i.e. in Denmark a jobless couple has 0.758 times the odds of conceiving of
#   a Danish dual equal-earner couple that is the same on every other variable.
#
# The baseline works the same way. "Constant" (0.133) is the odds of conceiving
# for the reference couple in Norway, and the "Denmark" country row (0.581)
# rescales it:
#
#   Danish baseline odds = Constant x Denmark = 0.133 x 0.581 = 0.0773
#
# The reference couple is in the first category of every variable below: woman
# aged 18-25, dual equal earners, household wages in the low tercile, low
# education, neither partner born abroad.
#
# Below, the Model 2 columns of Table A2 are typed in as
#   (variable, category, label in the paper, main row, Denmark row).
# A reference category has no row in the paper; it is written with 1.0 and 1.0.
# The woman's age has no "Denmark * age" row, so its Denmark row is also 1.0.
TABLE_A2_CONSTANT = 0.133
TABLE_A2_DENMARK = 0.581
TABLE_A2_MODEL_2 = [
    ("woman_age_group", "18-25", "Age female partner 18-25 (reference)", 1.0, 1.0),
    ("woman_age_group", "26-30", "Age female partner 26-30", 1.324, 1.0),
    ("woman_age_group", "31-35", "Age female partner 31-35", 1.219, 1.0),
    ("woman_age_group", "36-45", "Age female partner 36-45", 0.281, 1.0),
    ("earner_constellation", "dual_equal_earners", "Dual breadwinner, equal earners (reference)", 1.0, 1.0),
    ("earner_constellation", "both_jobless", "Jobless couple", 0.670, 1.132),
    ("earner_constellation", "male_works", "Male breadwinner", 0.951, 0.936),
    ("earner_constellation", "female_works", "Female breadwinner", 0.799, 1.041),
    ("earner_constellation", "dual_male_main_earner", "Dual breadwinner, male main earner", 0.864, 0.919),
    ("earner_constellation", "dual_female_main_earner", "Dual breadwinner, female main earner", 0.921, 1.107),
    ("household_wage_tercile", "low", "Household wages low (reference)", 1.0, 1.0),
    ("household_wage_tercile", "middle", "Household wages middle", 1.349, 1.277),
    ("household_wage_tercile", "high", "Household wages high", 1.890, 1.341),
    ("woman_education", "low", "Education female partner low (reference)", 1.0, 1.0),
    ("woman_education", "enrolled", "Enrolled in education", 0.542, 1.470),
    ("woman_education", "high", "Education female partner high", 1.197, 1.141),
    ("woman_education", "unknown", "Education female partner unknown", 1.113, 0.897),
    ("either_partner_foreign", 0, "Neither partner born abroad (reference)", 1.0, 1.0),
    ("either_partner_foreign", 1, "Foreign country of birth (either partner)", 1.010, 0.840),
]

# The Danish numbers, as stage 1 (to generate births) and stage 3 (to check
# the estimates) use them.
DANISH_BASELINE_ODDS = TABLE_A2_CONSTANT * TABLE_A2_DENMARK
DANISH_ODDS_RATIOS = {}
for _variable, _category, _label, _main_row, _denmark_row in TABLE_A2_MODEL_2:
    DANISH_ODDS_RATIOS.setdefault(_variable, {})[_category] = _main_row * _denmark_row


def danish_table_a2():
    """Table A2, Model 2, with the Norwegian reference folded in: one row per
    category, giving the Danish odds ratio and what it means as a probability."""
    # From odds to probability:  probability = odds / (1 + odds).
    # The Danish reference couple: 0.0773 / 1.0773 = 7.2% per year.
    def probability(odds):
        return odds / (1 + odds)

    rows = [{"variable": "baseline", "category": "", "label_in_paper": "Constant x Denmark",
             "compared_with": "(not a ratio: the odds of the Danish reference couple)",
             "danish_odds_ratio": round(DANISH_BASELINE_ODDS, 4),
             "yearly_probability_for_reference_couple": round(probability(DANISH_BASELINE_ODDS), 4)}]
    for variable, category, label, _, _ in TABLE_A2_MODEL_2:
        odds_ratio = DANISH_ODDS_RATIOS[variable][category]
        reference_label = next(label for v, _, label, _, _ in TABLE_A2_MODEL_2 if v == variable)
        # An odds ratio has no single probability equivalent: the same ratio
        # changes the probability more for a couple that starts from a higher
        # one. So the probability shown is for ONE fixed couple: the Danish
        # reference couple with only this trait changed.
        #   its odds        = Danish baseline odds x Danish odds ratio
        #   its probability = odds / (1 + odds)
        # e.g. jobless: 0.0773 x 0.758 = 0.0586 -> 5.5%, against 7.2% for equal earners.
        rows.append({"variable": variable, "category": category, "label_in_paper": label,
                     "compared_with": reference_label.replace(" (reference)", ""),
                     "danish_odds_ratio": round(odds_ratio, 3),
                     "yearly_probability_for_reference_couple":
                         round(probability(DANISH_BASELINE_ODDS * odds_ratio), 4)})
    return pd.DataFrame(rows)


def save_danish_table_a2():
    """Write the Danish table to the data folder. Needs no download."""
    DST_DATA_FOLDER.mkdir(exist_ok=True)
    danish_table_a2().to_csv(DST_DATA_FOLDER / "paper_table_a2_model2_denmark.csv", index=False)


def download(table, **selection):
    """One StatBank table as a DataFrame: a column per variable, the figure in `value`.

    `selection` names the values wanted for each of the table's variables, using
    StatBank's own codes (for example KØN="2" for women, Tid=2010 for the year).
    """
    request = {"table": table, "format": "CSV", "lang": "en", "valuePresentation": "Code",
               "variables": [{"code": code, "values": [str(v) for v in np.atleast_1d(values)]}
                             for code, values in selection.items()]}
    reply = urllib.request.urlopen(urllib.request.Request(
        "https://api.statbank.dk/v1/data", json.dumps(request).encode(),
        {"Content-Type": "application/json"}), timeout=180)
    table = pd.read_csv(io.StringIO(reply.read().decode("utf-8-sig")), sep=";", dtype=str)
    table.columns = [column.lower() for column in table.columns]
    table = table.rename(columns={"indhold": "value", "tid": "year", "køn": "sex", "koen": "sex",
                                  "alder": "age", "alder1": "age"})
    table["value"] = pd.to_numeric(table["value"].str.replace(",", "."), errors="coerce")
    if "sex" in table:
        table["sex"] = table["sex"].map(SEX_NAMES)
    return table


def spread_over_single_ages(table, value_columns):
    """Turn a table by age group ("25-29", "-19", "60-", or a single age "23") into one
    row per single age. Ages outside every group get the value of the nearest group."""
    def limits(group):
        if "-" not in group:
            return int(group), int(group)
        low, _, high = group.partition("-")
        return int(low or 0), int(high or 200)
    rows = []
    for keys, groups in table.groupby(["year", "sex"]):
        groups = groups.assign(low=[limits(g)[0] for g in groups.age],
                               high=[limits(g)[1] for g in groups.age]).sort_values("low")
        for age in range(YOUNGEST_AGE, OLDEST_AGE + 1):
            inside = groups[(groups.low <= age) & (groups.high >= age)]
            group = inside.iloc[0] if len(inside) else groups.iloc[-1 if age > groups.high.max() else 0]
            rows.append({"year": int(keys[0]), "sex": keys[1], "age": age,
                         **{column: group[column] for column in value_columns}})
    return pd.DataFrame(rows)


def women_in_start_year():
    """women_2010.csv - per age on 1 January 2010: how many women there are, how
    many are childless, and what share lives with a partner."""
    ages = list(range(8, 55))

    # TABLE FOLK1E "Population at the first day of the quarter".
    # Women (KØN 2), all of Denmark (OMRÅDE 000), all ancestries (HERKOMST TOT),
    # by single age, first quarter of 2010.
    women = download("FOLK1E", OMRÅDE="000", KØN="2", ALDER=ages, HERKOMST="TOT",
                     Tid=f"{START_YEAR}K1")
    women = women.assign(age=women.age.astype(int)).set_index("age")["value"].rename("all_women")

    # TABLE BARNLOS "Childless per 1,000".
    # Women (KØN 2) by single age, 2010. The table starts at age 20, so women
    # under 20 are all counted as childless.
    # Number of childless women = population x childless per 1,000 / 1,000.
    childless_per_1000 = download("BARNLOS", KØN="2", ALDER=[a for a in ages if a >= 20],
                                  Tid=START_YEAR)
    childless_per_1000 = childless_per_1000.assign(
        age=childless_per_1000.age.astype(int)).set_index("age")["value"]
    childless_women = (women * childless_per_1000.reindex(women.index).fillna(1000) / 1000).round()

    # TABLE FAM100N "Adults 1 January".
    # Women (KØN 2) by single age, family type (FAMTYP) and number of children
    # living in the family (ANTBRNF), 2010. A woman lives with a partner unless
    # her family type is single man (M) or single woman (K).
    families = download("FAM100N", KØN="2", FAMTYP="*", ANTBRNF="*",
                        ALDER=[a for a in ages if a >= 18], Tid=START_YEAR)
    families = families.assign(age=families.age.astype(int))
    in_couple = families[~families.famtyp.isin(["M", "K"])]
    all_in_couple = in_couple.groupby("age")["value"].sum()
    # For the share of childless women with a partner, both counts use this
    # table's definition of childless (no children living in the family), so
    # that numerator and denominator are consistent with each other.
    childless_in_couple = in_couple[in_couple.antbrnf == "0"].groupby("age")["value"].sum()
    without_children_in_family = women - families[families.antbrnf != "0"].groupby("age")["value"].sum()

    return pd.DataFrame({
        "birth_year": START_YEAR - 1 - women.index,
        "all_women": women.astype(int),
        "childless_women": childless_women.astype(int),
        "share_of_all_women_living_with_partner":
            (all_in_couple / women).reindex(women.index).fillna(0).round(4),
        "share_of_childless_women_living_with_partner":
            (childless_in_couple / without_children_in_family).reindex(women.index).fillna(0).round(4),
    }).rename_axis("age_on_1_january").reset_index()


def immigrant_share_in_start_year():
    """immigrant_share_2010.csv - share of immigrants (people born abroad) by sex
    and single age, 1 January 2010."""
    # TABLE FOLK1E "Population at the first day of the quarter".
    # Men and women (KØN 1, 2), all of Denmark, by single age, first quarter of
    # 2010: everyone (HERKOMST TOT) and immigrants from western and non-western
    # countries (HERKOMST 24, 25).
    population = download("FOLK1E", OMRÅDE="000", KØN=["1", "2"], ALDER=list(range(8, OLDEST_AGE + 1)),
                          HERKOMST=["TOT", "24", "25"], Tid=f"{START_YEAR}K1")
    population["age"] = population.age.astype(int)
    population["group"] = np.where(population.herkomst == "TOT", "everyone", "immigrants")
    counts = population.groupby(["sex", "age", "group"])["value"].sum().unstack()
    return (counts.immigrants / counts.everyone).round(4).rename("share_immigrant").reset_index()


def yearly_rates():
    """yearly_rates.csv - per year 2010-2019, sex and single age 18-65: employment
    rate, education shares, average wage, wage spread and enrolment rate."""
    years = list(YEARS)

    # TABLE RAS200 "Activity and employment rates (end November)".
    # Employment rate in per cent (BEREGNING BFK), men and women, all of Denmark
    # (OMRÅDE 000), all ancestries (HERKOMST 00), by age group, each year.
    employment = download("RAS200", OMRÅDE="000", HERKOMST="00", KØN=["M", "K"], BEREGNING="BFK",
                          ALDER=["18-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                                 "50-54", "55-59", "60-64"], Tid=years)
    employment["employment_rate"] = (employment.value / 100).round(4)
    rates = spread_over_single_ages(employment, ["employment_rate"])

    # TABLE HFUDD11 "Educational attainment (15-69 years)".
    # Number of people by highest completed education (HFUDD), men and women, all
    # of Denmark (BOPOMR 000), all ancestries (HERKOMST TOT), by age group, each
    # year. "High" = short-cycle higher education and above (H40, H50, H60, H70,
    # H80); "unknown" = not stated (H90); TOT = everyone.
    attainment = download("HFUDD11", BOPOMR="000", HERKOMST="TOT", KØN=["M", "K"],
                          HFUDD=["TOT", "H40", "H50", "H60", "H70", "H80", "H90"],
                          ALDER=["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                                 "50-54", "55-59", "60-64", "65-69"], Tid=years)
    attainment["level"] = attainment.hfudd.map(
        lambda code: {"TOT": "everyone", "H90": "unknown"}.get(code, "high"))
    counts = attainment.groupby(["year", "sex", "age", "level"])["value"].sum().unstack().reset_index()
    counts["share_high_education"] = (counts.high / counts.everyone).round(4)
    counts["share_education_unknown"] = (counts.unknown / counts.everyone).round(4)
    rates = rates.merge(spread_over_single_ages(
        counts, ["share_high_education", "share_education_unknown"]), on=["year", "sex", "age"])

    # TABLE INDKP111 "Income".
    # Average income in DKK of people who have that type of income (ENHED 121),
    # for wages and salaries (INDKOMSTTYPE 115), men and women, all of Denmark
    # (REGLAND 000), by age group, each year.
    wages = download("INDKP111", REGLAND="000", ENHED="121", KOEN=["M", "K"], INDKOMSTTYPE="115",
                     ALDER1=["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                             "50-54", "55-59", "60-64", "65-69"], Tid=years)
    wages["average_wage"] = wages.value.round()
    rates = rates.merge(spread_over_single_ages(wages, ["average_wage"]), on=["year", "sex", "age"])

    # TABLE LONS50 "Earnings".
    # Lower and upper quartile of standardised hourly earnings (LØNMÅL NEDREST,
    # OVREST), men and women, all sectors (SEKTOR 1000), all forms of pay (AFLOEN
    # TIFA), all employees (LONGRP LTOT), by age group. For a log-normal
    # distribution, sd of log wage = ln(upper quartile / lower quartile) / 1.349.
    # The table starts in 2013; 2010-2012 use the 2013 figure.
    quartiles = download("LONS50", SEKTOR="1000", AFLOEN="TIFA", LONGRP="LTOT", KØN=["M", "K"],
                         LØNMÅL=["NEDREST", "OVREST"],
                         ALDER1=["-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49",
                                 "50-54", "55-59", "60-"],
                         Tid=[y for y in years if y >= 2013])
    quartiles = quartiles.pivot_table(index=["year", "sex", "age"], columns="lønmål",
                                      values="value").reset_index()
    quartiles["wage_log_sd"] = (np.log(quartiles.OVREST / quartiles.NEDREST) / 1.349).round(4)
    spread = spread_over_single_ages(quartiles, ["wage_log_sd"])
    first_year = spread[spread.year == spread.year.min()]
    spread = pd.concat([first_year.assign(year=y) for y in years if y < spread.year.min()] + [spread])
    rates = rates.merge(spread, on=["year", "sex", "age"])

    # TABLE UDDAKT10 "Educational activity".
    # Number of students on 1 October (FSTATUS B), all educations (UDDANNELSE
    # TOT), men and women, all of Denmark (BOPOMR 000), by single age 18-39 and
    # "40 and over", each year.
    students = download("UDDAKT10", BOPOMR="000", UDDANNELSE="TOT", KØN=["M", "K"], FSTATUS="B",
                        ALDER=list(range(YOUNGEST_AGE, 40)) + ["40-"], Tid=years)
    # TABLE FOLK1E "Population at the first day of the quarter".
    # Men and women, all of Denmark, by single age 18-65, fourth quarter of each
    # year. Enrolment rate = students / population of the same age; ages 40-65
    # are added up to match UDDAKT10's "40 and over".
    population = download("FOLK1E", OMRÅDE="000", KØN=["1", "2"], HERKOMST="TOT",
                          ALDER=list(range(YOUNGEST_AGE, OLDEST_AGE + 1)),
                          Tid=[f"{y}K4" for y in years])
    population["year"] = population.year.str[:4]
    population["age"] = population.age.astype(int).clip(upper=40).astype(str).replace("40", "40-")
    population = population.groupby(["year", "sex", "age"])["value"].sum().rename("population")
    students = students.join(population, on=["year", "sex", "age"])
    students["enrolment_rate"] = (students.value / students.population).round(4)
    rates = rates.merge(spread_over_single_ages(students, ["enrolment_rate"]), on=["year", "sex", "age"])

    return rates.sort_values(["year", "sex", "age"])


# ---- charts: a visual check of what was downloaded ------------------------------
# Each chart is saved in the data folder under the name of the StatBank table(s)
# its figures come from, so it can be compared with the table on statistikbanken.dk.
WOMEN_COLOUR, SECOND_COLOUR = "#2a78d6", "#eb6834"
# One shade of blue per year, light (first year) to dark (last year).
YEAR_COLOURS = ["#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
AS_PERCENT = FuncFormatter(lambda value, _: f"{value:.0%}")
AS_NUMBER = FuncFormatter(lambda value, _: f"{value:,.0f}")


def style(axis, title, x_label, y_format):
    """The look shared by every chart: left-aligned title, light horizontal grid,
    no top or right border, y axis starting at zero."""
    axis.set_title(title, fontsize=10, loc="left", color="#0b0b0b")
    axis.set_xlabel(x_label, fontsize=9, color="#52514e")
    axis.yaxis.set_major_formatter(y_format)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#e1e0d9", lw=0.6, zorder=0)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#c3c2b7")
    axis.tick_params(colors="#52514e", labelsize=8.5)


def chart_two_lines(file_name, title, x_label, y_format, x, first, second):
    """One chart with two lines. `first` and `second` are (label, values)."""
    figure, axis = plt.subplots(figsize=(8, 4.2), dpi=130)
    for (label, values), colour in ((first, WOMEN_COLOUR), (second, SECOND_COLOUR)):
        axis.plot(x, values, color=colour, lw=2, label=label, zorder=2)
    style(axis, title, x_label, y_format)
    axis.legend(frameon=False, fontsize=8.5)
    figure.tight_layout()
    figure.savefig(DST_DATA_FOLDER / file_name)
    plt.close(figure)


def chart_by_year(file_name, title, y_format, rates, column):
    """One panel for women and one for men; in each, one line per year by age."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130, sharey=True)
    years = sorted(rates.year.unique())
    for axis, (sex, sex_label) in zip(axes, (("F", "Women"), ("M", "Men"))):
        for year, colour in zip(years, YEAR_COLOURS):
            one_year = rates[(rates.sex == sex) & (rates.year == year)]
            # Drawn as steps: most tables are by age group, so the figure is flat within a group.
            axis.step(one_year.age, one_year[column], where="mid", color=colour, lw=1.5, zorder=2,
                      label=year if year in (years[0], years[-1]) else None)
        style(axis, f"{title} - {sex_label}", "Age", y_format)
    axes[0].legend(frameon=False, fontsize=8.5, title="One line per year", title_fontsize=8.5)
    figure.tight_layout()
    figure.savefig(DST_DATA_FOLDER / file_name)
    plt.close(figure)


def save_charts(women, immigrants, rates):
    """Draw every downloaded figure. File names start with the StatBank table name."""
    chart_two_lines("FOLK1E_BARNLOS_women_by_age_2010.png",
                    "Women on 1 January 2010, by age (FOLK1E; childless = FOLK1E x BARNLOS)",
                    "Age on 1 January 2010", AS_NUMBER, women.age_on_1_january,
                    ("All women", women.all_women), ("Childless women", women.childless_women))
    chart_two_lines("FAM100N_share_living_with_partner_2010.png",
                    "Share of women living with a partner, 1 January 2010 (FAM100N)",
                    "Age on 1 January 2010", AS_PERCENT, women.age_on_1_january,
                    ("All women", women.share_of_all_women_living_with_partner),
                    ("Childless women", women.share_of_childless_women_living_with_partner))
    by_sex = immigrants.pivot(index="age", columns="sex", values="share_immigrant")
    chart_two_lines("FOLK1E_share_immigrant_2010.png",
                    "Share of immigrants, 1 January 2010 (FOLK1E)",
                    "Age on 1 January 2010", AS_PERCENT, by_sex.index,
                    ("Women", by_sex["F"]), ("Men", by_sex["M"]))
    for file_name, title, y_format, column in (
            ("RAS200_employment_rate.png", "Employment rate (RAS200)", AS_PERCENT, "employment_rate"),
            ("HFUDD11_share_high_education.png", "Share with higher education (HFUDD11)",
             AS_PERCENT, "share_high_education"),
            ("HFUDD11_share_education_unknown.png", "Share with education not stated (HFUDD11)",
             AS_PERCENT, "share_education_unknown"),
            ("INDKP111_average_wage.png", "Average wage of people with wage income, DKK (INDKP111)",
             AS_NUMBER, "average_wage"),
            ("LONS50_wage_log_sd.png", "Spread of wages: sd of log hourly earnings (LONS50)",
             FuncFormatter(lambda value, _: f"{value:.2f}"), "wage_log_sd"),
            ("UDDAKT10_FOLK1E_enrolment_rate.png", "Share enrolled in education (UDDAKT10 / FOLK1E)",
             AS_PERCENT, "enrolment_rate")):
        chart_by_year(file_name, title, y_format, rates, column)


def download_all():
    DST_DATA_FOLDER.mkdir(exist_ok=True)
    save_danish_table_a2()
    print("  paper_table_a2_model2_denmark.csv: the paper's Danish odds ratios")
    tables = {}
    for file_name, make_table in (("women_2010.csv", women_in_start_year),
                                  ("immigrant_share_2010.csv", immigrant_share_in_start_year),
                                  ("yearly_rates.csv", yearly_rates)):
        tables[file_name] = make_table()
        tables[file_name].to_csv(DST_DATA_FOLDER / file_name, index=False)
        print(f"  {file_name}: {len(tables[file_name])} rows")
    save_charts(*tables.values())
    print(f"  {len(list(DST_DATA_FOLDER.glob('*.png')))} charts saved in {DST_DATA_FOLDER.name}/")


if __name__ == "__main__":
    print("Stage 0: downloading Statistics Denmark figures ...")
    download_all()
