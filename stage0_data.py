"""
STAGE 0 - Download every Statistics Denmark figure that stage 1 draws from, and
save them as three small tables in 0data/. Nothing here is simulated: these
are published counts and rates.
The tables are kept in the repository, so this stage only needs to run again
to refresh them.

Source: StatBank (statistikbanken.dk) through its public API, no login needed.
Every download below is preceded by a comment naming the StatBank table, which
can be opened at statistikbanken.dk/<table name>.

The paper's Danish odds ratios (0data/paper_table_a2_model2_denmark.csv) are
not made here: that file is a ready-made input, read in common.py.

Statistics Denmark tables used, and the file each ends up in:
  women_2010.csv              FOLK1E, BARNLOS, FAM100N
  immigrant_share_2010.csv    FOLK1E
  yearly_rates.csv            RAS200, HFUDD11, INDKP111, LONS50, UDDAKT10, FOLK1E
"""
import io
import json
import urllib.request

import numpy as np
import pandas as pd

from common import DST_DATA_FOLDER, OLDEST_AGE, YEARS, YOUNGEST_AGE

START_YEAR = min(YEARS)                 # traits that are drawn once are taken from this year;
                                        # traits that are drawn yearly from each of YEARS
SEX_NAMES = {"K": "F", "2": "F", "M": "M", "1": "M"}     # StatBank's codes -> F / M


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


def download_all():
    DST_DATA_FOLDER.mkdir(exist_ok=True)
    for file_name, make_table in (("women_2010.csv", women_in_start_year),
                                  ("immigrant_share_2010.csv", immigrant_share_in_start_year),
                                  ("yearly_rates.csv", yearly_rates)):
        table = make_table()
        table.to_csv(DST_DATA_FOLDER / file_name, index=False)
        print(f"  {file_name}: {len(table)} rows")


if __name__ == "__main__":
    print("Stage 0: downloading Statistics Denmark figures ...")
    download_all()
