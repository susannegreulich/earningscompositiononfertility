# Register simulation: couples' employment, wages and first births

A synthetic rebuild, for Denmark, of the data pipeline behind Marynissen,
Wood, Fallesen, Mikolai, Hellstrand & Bergsvik (2026), *His and/or hers?*
(ROCKWOOL Foundation Study Paper 298).

The real study used restricted registers. Each country team built couple-year
files internally and shared only collapsed count tables. This project
simulates all three data levels, so each one can be inspected and changed.

## The logic

```
stage 0  download Statistics Denmark figures dst_data/*.csv
   │
stage 1  simulate person-level registers    output/1_simulated_register_*.csv
   │
stage 2  build the couple-year file         output/2_couple_years.csv
   │
stage 3  collapse to a count table          output/3_event_occurrence_table.csv
   │
stage 4  fit the paper's model              output/4_*
```

0. **Download.** Fetch every Statistics Denmark figure the simulation needs
   and save it in `dst_data/`: counts of childless women, and rates of
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
   - *Generated from the paper's model:* whether a childless woman with a
     partner conceives that year. The probability comes straight from the
     paper's logit equation (eq. 1) with the Danish estimates in its Table A2:
     the couple's employment-wage constellation, her age and education,
     household wages and foreign origin.

   The result is five files imitating the Danish population, employment,
   income, education and birth registers.
2. **Build couple-years.** Using only those registers, keep each year in which
   a woman aged 18–45 lives with a partner and is still childless. Classify
   the couple (both jobless, only he works, only she works, or both work with
   equal or unequal wages) and record whether a first child was conceived
   that year.
3. **Collapse.** Count first births and couple-years for every combination of
   the model variables. This is all the paper's authors received.
4. **Estimate.** Fit the paper's logit model on the counts and compare
   the odds ratios with Table A2.

Each stage reads only the output of the stage before it: stage 1 reads
`dst_data/`, stage 2 the registers, stage 3 the couple-years, stage 4 the counts.

**Why this is useful:** the odds ratios that generate the births in stage 1
are known, so stage 4 can be checked against them. If stages 2 to 4 are coded
correctly, the estimates reproduce Table A2.

Only the paper's Model 2 is recreated: the model with controls for household
wages, education and foreign origin. The paper also reports a Model 1 without
those controls, but its estimates depend on how traits are distributed in the
sample, so they cannot be reproduced without also reproducing the paper's
sample.

## Run

```bash
pip install -r requirements.txt
python run_all.py          # all stages, about 25 seconds
python run_all.py 2 3 4    # only the later stages
python run_all.py 0        # download the Statistics Denmark figures again
```

## Where things are

| File | Contents |
|---|---|
| `stage0_download_dst_data.py` | downloads the Statistics Denmark figures into `dst_data/`, with a comment naming the table behind each download |
| `stage1_simulate_registers.py` | the Table A2 odds ratios, the one assumption, and the simulation |
| `stage2_build_couple_years.py` | who is in the sample, and the derived variables |
| `stage3_aggregate.py` | the count table and an optional small-cell suppression rule |
| `stage4_analyse.py` | the model and the comparison of odds ratios with Table A2 |
| `common.py` | file locations, category lists, the classification rules |
| `dst_data/` | the three tables of Statistics Denmark figures that stage 1 draws from |

To change a setting, edit the constants at the top of the stage file and rerun.
To change a Statistics Denmark figure, edit the table in `dst_data/`.

## Where the numbers come from

- **First births:** the paper's logit equation with the Danish Model 2 odds
  ratios in Table A2. This is the only part that must match the paper.
- **Everything else:** Statistics Denmark's StatBank (statistikbanken.dk),
  downloaded by stage 0 through the public API into three small tables in
  `dst_data/`.

  | Trait | StatBank table | Year | File in `dst_data/` |
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
A1). The model contains every trait the births depend on, so it recovers Table
A2 whatever the mix of ages, constellations or education levels is.

## Results with the default parameters

- **The estimates reproduce Table A2.** The paper's Danish odds ratio lies inside
  the 95% confidence interval for all 15 estimates. For the constellations,
  relative to dual equal earners:

  | Constellation | Estimated | Table A2 |
  |---|---|---|
  | Both jobless | 0.75 | 0.76 |
  | Only he works | 0.91 | 0.89 |
  | Only she works | 0.86 | 0.83 |
  | Both work, he earns more | 0.80 | 0.79 |
  | Both work, she earns more | 1.05 | 1.02 |

- **Size.** With the default 20% of Denmark's childless women: 152,000 women,
  78,000 partners, 367,000 couple-years and 39,000 first births.

## Limits

- A person's traits are independent of each other: employment does not depend
  on education, and partners' wages are unrelated.
- Couples never break up, and a woman has at most one partner.
- Nobody enters after 2010: women who immigrate later are not added.
- First births only; no mortality; men exist only once they live with a simulated woman.
