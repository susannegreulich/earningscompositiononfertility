# Register simulation: couples' employment, wages and first births

A synthetic rebuild, for Denmark, of the data pipeline behind Marynissen,
Wood, Fallesen, Mikolai, Hellstrand & Bergsvik (2026), *His and/or hers?*
(ROCKWOOL Foundation Study Paper 298).
The study is published at
<https://rockwoolfonden.dk/udgivelser/his-and-or-hers-a-cross-national-perspective-on-the-gendered-nature-of-couples-employment-preconditions-for-parenthood/>.

The real study used restricted registers. Each country team built couple-year
files internally and shared only collapsed count tables. This project
simulates all three data levels, so each one can be inspected and changed.

## The logic

```
stage 0  download Statistics Denmark figures   0data/
   │
stage 1  simulate person-level registers       1simulated_register_data/
   │
stage 2  build couple-years, then count them    2processed_data/
   │
stage 3  fit the paper's model                 3estimation/
```

0. **Download.** Fetch every Statistics Denmark figure the simulation needs
   and save it in `0data/`: counts of childless women, and rates of
   employment, enrolment, education, wages and living with a partner. Nothing
   in this step is simulated.
1. **Simulate registers.** The population is the women born 1965–2001 who
   were childless at the start of 2010, counted from Statistics Denmark, plus
   the partners they live with. They are followed through 2010–2019.
   - *Drawn once and kept:* birth year, origin, education, and the year she
     moves in with a partner (if ever).
   - *Drawn every year:* employed, enrolled and wage, each from that year's
     Statistics Denmark figure for the person's sex and age, independently of
     every other trait.
   - *Generated from the paper's model, using the probability function from the paper* whether a childless woman with a
     partner conceives that year. The probability comes straight from the
     paper's logit equation (eq. 1) with added controls (model 2) with the Danish estimates in its Table A2: the couple's employment-wage constellation, her age and education,
     household wages and foreign origin.

   The result is five files imitating the Danish population, employment,
   income, education and birth registers.
2. **Process.** Two steps in one script.
   - *Build couple-years.* Using only those registers, keep each year in which
     a woman aged 18–45 lives with a partner and is still childless. Classify
     the couple (both jobless, only he works, only she works, or both work
     with equal or unequal wages) and record whether a first child was
     conceived that year.
   - *Count.* Count first births and couple-years for every combination of
     the model variables. This is all the paper's authors received.
3. **Estimate.** Fit the paper's logit model on the counts and compare
   the odds ratios with Table A2.

Each stage reads only the output of the stage before it: stage 1 reads
`0data/`, stage 2 the registers, stage 3 the counts.
All four folders are kept in the repository, so the registers, the couple-years
and the results can be inspected without running anything; running the
pipeline rewrites them.

**Why this is useful:** the odds ratios that generate the births in stage 1
are known (the simulation of birth expressedly used the paper's parameter estimates), 
so stage 3 estimation results can be checked against them. If stages 2 and 3 are coded
correctly, the estimates reproduce Table A2.

Only the paper's Model 2 is recreated: the model with controls for household
wages, education and foreign origin. The paper also reports a Model 1 without
those controls, but its estimates depend on how traits are distributed in the
sample, so they cannot be reproduced without also reproducing the paper's
sample, which we cannot because we do not have microdata, only DST's aggregate averages. 

## Run

```bash
pip install -r requirements.txt
python run_all.py          # all stages, about 25 seconds
python run_all.py 2 3      # only the later stages
python run_all.py 0        # download the Statistics Denmark figures again
```

## Where things are

| File | Contents |
|---|---|
| `stage0_data.py` | downloads the Statistics Denmark figures into `0data/`, with a comment naming the table behind each download; also holds the paper's Table A2 and computes the Danish odds ratios from it |
| `stage1_simulate_registers.py` | the one assumption and the simulation |
| `stage2_process_data.py` | who is in the sample, the derived variables, and the count table with an optional small-cell suppression rule |
| `stage3_analyse.py` | the model, the comparison of odds ratios with Table A2 as a table and as charts, and the odds for each constellation |
| `common.py` | file locations, category lists, the classification rules |
| `0data/` | the three tables of Statistics Denmark figures that stage 1 draws from, a chart of each figure named after its StatBank table, and `paper_table_a2_model2_denmark.csv`, the paper's odds ratios for Denmark |
| `1simulated_register_data/` | the five simulated registers: `bef`, `ras`, `ind`, `udda`, `fod` |
| `2processed_data/` | the couple-year file and the count table |
| `3estimation/` | the estimated odds ratios next to the paper's Danish ones, as a table and as charts by constellation, household wages and age; and the odds themselves for each constellation, written as fractions such as "1 to 13" |

To change a setting, edit the constants at the top of the stage file and rerun.
To change a Statistics Denmark figure, edit the table in `0data/`.

## Where the numbers come from

- **First births:** the paper's logit equation with the Danish Model 2 odds
  ratios from Table A2. This is the only part that must match the paper.
  Table A2 uses Norway as the reference country, so it prints no Danish odds
  ratio directly: each one is the product of a main row and a "Denmark * …"
  row (for jobless couples, 0.670 × 1.132 = 0.758). Stage 0 holds the table,
  explains the calculation, and writes the Danish values to
  `0data/paper_table_a2_model2_denmark.csv`.
- **Everything else:** Statistics Denmark's StatBank (statistikbanken.dk),
  downloaded by stage 0 through the public API into three small tables in
  `0data/`.

  | Trait | StatBank table | Year | File in `0data/` |
  |---|---|---|---|
  | How many childless women, per birth year | BARNLOS, FOLK1E | 2010 | `women_2010.csv` |
  | Living with a partner, by age | FAM100N | 2010 | `women_2010.csv` |
  | Born abroad | FOLK1E | 2010 | `immigrant_share_2010.csv` |
  | Partner's age | FOD11, VIE825 | 2010 | typed into stage 0, with its source |
  | Education level | HFUDD11 | each year 2010–2019 | `yearly_rates.csv` |
  | Employed | RAS200 | each year | `yearly_rates.csv` |
  | Enrolled | UDDAKT10, FOLK1E | each year | `yearly_rates.csv` |
  | Wage: average | INDKP111 | each year | `yearly_rates.csv` |
  | Wage: spread | LONS50 | each year from 2013 | `yearly_rates.csv` |

- **One assumption:** how often a person keeps last year's position in the
  employment, enrolment and wage distributions (80%). Without it, each
  person's status would jump at random from year to year. It does not change
  the share employed, enrolled or at a given wage in any year.

The simulated sample does not have to look like the paper's sample (its Table
A1). The model contains every trait the births depend on, and is simulated using the
parameter estimates as coefficients, so it recovers Table
A2 whatever the mix of ages, constellations or education levels is in the simulated register data.

## Results with the default parameters

- **The estimates reproduce Table A2.** The paper's Danish odds ratio lies inside
  the 95% confidence interval for all 15 estimates. For the constellations,
  relative to dual equal earners:

  | Constellation | Estimated | Table A2 |
  |---|---|---|
  | Both jobless | 0.78 | 0.76 |
  | Only he works | 0.91 | 0.89 |
  | Only she works | 0.84 | 0.83 |
  | Both work, he earns more | 0.80 | 0.79 |
  | Both work, she earns more | 1.02 | 1.02 |

- **Size.** With the default 20% of Denmark's childless women: 152,000 women,
  78,000 partners, 367,000 couple-years and 39,000 first births.

## Interpreting results

**Odds, intuitively.** Odds compare how often something happens with how often it doesn't. Take 1,000 Danish reference couples in one year: about 72 conceive and 928 don't. Their odds are 72 to 928, or roughly 1 to 13: for every couple that conceives, about 13 don't.

**The odds ratio** then says how that "1 to 13" shifts for another group. For jobless couples it is 0.78. Multiply the first number: 0.78 to 13, or about 1 to 17. So where equal-earner couples have one conception for every 13 without, jobless couples have one for every 17.

In this study the event is rare enough (under about 20% a year) that you can read an odds ratio as almost the same thing as a ratio of probabilities. So as a working rule here:

| Odds ratio | Read it as |
|---|---|
| 0.78 | roughly 22% fewer conceptions than equal earners |
| 0.91 | roughly 9% fewer |
| 1.02 | about the same |

"Fewer" always means compared with equal-earner couples who are otherwise identical: same age group, same household wages, same education, same origin.

**Reading the figure.**

- The line at 1.0 is the equal-earner couples. Every bar is measured against them, which is why they have no bar.
- A bar's height is the estimated odds ratio. Below the line means fewer conceptions than equal earners; above means more.
- The thin black line on each bar is the 95% confidence interval: the range of values the data are compatible with.
- The diamond is the paper's Danish odds ratio from Table A2.

## Limits: Assumptions made during simulation of register data

- A person's traits are independent of each other: employment does not depend
  on education, and partners' wages are unrelated.
- Couples never break up, and a woman has at most one partner.
- Nobody enters after 2010: women who immigrate later are not added.
- First births only; no mortality; men exist only once they live with a simulated woman.
