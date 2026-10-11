"""
STAGE 3 - Collapse the couple-year file into the event-occurrence table: one
row per combination of the model variables, with the number of first births
and the number of couple-years. This is the format the paper's authors
received from each country team. Uses ONLY the stage 2 file.
"""
import pandas as pd

from common import COUPLE_YEAR_FILE, EVENT_TABLE_FILE, PROCESSED_DATA_FOLDER

# =============================================================================
# PARAMETERS
# =============================================================================
# The model variables that define a cell.
CELL_VARIABLES = ["woman_age_group", "earner_constellation", "household_wage_tercile",
                  "woman_education", "either_partner_foreign"]
# Statistical-office disclosure rule: cells with fewer couple-years are dropped
# (0 = keep everything). Dropping cells changes the estimates - try it.
MIN_COUPLE_YEARS_PER_CELL = 0
# =============================================================================


def aggregate():
    couple_years = pd.read_csv(COUPLE_YEAR_FILE)

    # Group the couple-years by every model variable. That is at most
    # 4 x 6 x 3 x 4 x 2 = 576 cells; combinations that never occur produce no
    # row. For each cell: first_births = number of couple-years with the event,
    # couple_years = number of rows (the exposure). The crude rate of a cell is
    # first_births / couple_years, and a logit fitted to (first_births out of
    # couple_years) gives exactly the same estimates as a logit on the stage 2
    # rows.
    cells = (couple_years.groupby(CELL_VARIABLES)["conceived_first_child_that_year"]
             .agg(first_births="sum", couple_years="size").reset_index())

    # Disclosure control: drop cells with too few couple-years.
    too_small = cells.couple_years < MIN_COUPLE_YEARS_PER_CELL
    suppressed_cells = cells[too_small]
    cells = cells[~too_small]

    PROCESSED_DATA_FOLDER.mkdir(exist_ok=True)
    cells.to_csv(EVENT_TABLE_FILE, index=False)
    print(f"  {EVENT_TABLE_FILE.name}: {len(cells)} cells "
          f"(of max 576; {len(suppressed_cells)} suppressed, "
          f"{suppressed_cells.couple_years.sum():,} couple-years lost)")
    return cells


if __name__ == "__main__":
    print("Stage 3: aggregating ...")
    aggregate()
