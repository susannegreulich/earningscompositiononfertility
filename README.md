# Register simulation: couples' employment, wages and first births

A synthetic rebuild, for Denmark, of the data pipeline behind Marynissen,
Wood, Fallesen, Mikolai, Hellstrand & Bergsvik (2026), *His and/or hers?*
(ROCKWOOL Foundation Study Paper 298).

The real study used restricted registers. Each country team built couple-year
files internally and shared only collapsed count tables. This project
simulates all three data levels, so each one can be inspected and changed.

## The logic

```
stage 1  simulate person-level registers    output/1_simulated_register_*.csv
   │
stage 2  build the couple-year file         output/2_couple_years.csv
   │
stage 3  collapse to a count table          output/3_event_occurrence_table.csv
   │
stage 4  fit the paper's models             output/4_*
```

1. **Simulate registers.** Women born 1965–2002 are followed yearly from age
   18 to 46. Each year they may leave or enter education, form or end a
   relationship, find or lose a job, and earn a wage; so may their partners.
   Childless women with a partner may conceive, with a probability that
   depends on the couple's employment-wage constellation and other traits.
   The result is five files imitating the Danish population, employment,
   income, education and birth registers.
2. **Build couple-years.** Using only those registers, keep each year in which
   a woman aged 18–45 lives with a partner and is still childless. Classify
   the couple (jobless, male or female breadwinner, or dual earners with
   equal or unequal wages) and record whether a first child was conceived
   that year.
3. **Collapse.** Count first births and couple-years for every combination of
   the model variables. This is all the paper's authors received.
4. **Estimate.** Fit the paper's two logit models on the counts and compute
   the effect of each constellation relative to dual equal earners.

Each stage reads only the output of the stage before it.

**Controlling for the income confounding variable:** the birth probabilities that drive stage 1 are known,
so stage 4 can be checked against the truth. Model 2 (with controls) should
recover it. Model 1 (age only) should not, because jobless and single-earner
couples also have lower household wages.

## Run

```bash
pip install -r requirements.txt
python run_all.py          # all stages, about 30 seconds
python run_all.py 2 3 4    # only the later stages
```

## Where things are

| File | Contents |
|---|---|
| `stage1_simulate_registers.py` | all behavioural parameters, with their sources, and the simulation |
| `stage2_build_couple_years.py` | derived variables |
| `stage3_aggregate.py` | the count table and an optional small-cell suppression rule |
| `stage4_analyse.py` | models, marginal effects, comparison with the paper |
| `common.py` | file locations, category lists, the constellation rule |

To change an assumption, edit the constants at the top of the stage file and rerun.

## Where the parameters come from

- **First-birth probabilities:** the paper's Danish Model 2 (Table A2).
- **Population, education, employment and wage levels:** Statistics Denmark's
  StatBank (statistikbanken.dk), averaged over 2010–2019, the paper's
  observation period. StatBank publishes levels (who is employed, enrolled,
  in a couple) but not yearly transitions, so many transition probabilities
  are set by hand to reproduce a published level.
- **Assumptions with no source:** wage dynamics, break-ups of childless
  couples, and selection into unions by employment. These are chosen so that
  the couple-year sample has the age mix and the earner constellations of the
  paper's Table A1.

## Results with the default parameters

- **The couple-year sample is close to the paper's** on the two things that
  matter most for the models:

  | Share of couple-years | Simulated | Paper |
  |---|---|---|
  | Women 18–25 / 26–30 / 31–35 / 36–45 | 36 / 30 / 16 / 18% | 37 / 31 / 15 / 18% |
  | Dual equal earners | 27% | 29% |
  | Dual, male / female main earner | 25 / 8% | 24 / 10% |
  | Male / female breadwinner | 20 / 12% | 17 / 11% |
  | Jobless | 8% | 9% |

- **Model 2 recovers the truth** within its 95% confidence interval for four
  of the five constellations. Female breadwinner is outside (−0.011 against a
  true −0.017).
- **Model 1 is close to the paper's Danish estimates** for the jobless (−0.088
  against −0.096) and the single-earner couples, less so for male main earners
  (−0.030 against −0.043), and not for female main earners (+0.009 against
  −0.034).
- **Two known mismatches.** Far more women are enrolled than in the paper (29%
  against 1.3%), which suggests the paper defines "enrolled" more narrowly
  than StatBank. And the simulation has too few mothers, so the share of all
  women living in a couple is too low after age 30.

  there are two, and both come from the paper. They are the same kind of model, a logit for whether a childless couple conceives a first child in a given year, and differ only in what they control for:

Model 1 uses the woman's age group and the couple's earner constellation.
Model 2 uses those two plus the household-wage tercile, the woman's education, and whether either partner is foreign-born.
Both are fitted in stage4_analyse.py on the stage 3 count table, and both report the effect of each constellation relative to dual equal earners.

The reason for having two is to separate a gross association from a net one. Take jobless couples: in Model 1 they are about 9 percentage points less likely to conceive than equal earners; in Model 2 only about 2. Most of the Model 1 gap arises because jobless couples also have low household wages, and low wages themselves go with fewer births. Once Model 2 holds wages, education and origin fixed, what remains is the part tied to the constellation itself.

In this project the comparison has an extra use. The birth probabilities in stage 1 are built from the constellation plus exactly the Model 2 controls, so Model 2 is the correctly specified model and should reproduce the values you put in. That is the check that stages 2 to 4 are coded correctly. Model 1 leaves those controls out, so it should differ from the truth, and the size of that difference shows how much of the raw pattern is composition.

## Limits

- All parameters are constant over time: no business cycle, no postponement.
- First births only; no migration or mortality; men exist only while partnered.
