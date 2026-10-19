# Register simulation: couples' employment, wages and first births

A synthetic rebuild, for Denmark, of the data pipeline behind Marynissen,
Wood, Fallesen, Mikolai, Hellstrand & Bergsvik (2026), *His and/or hers?*
(ROCKWOOL Foundation Study Paper 298).
The study is published at
<https://rockwoolfonden.dk/udgivelser/his-and-or-hers-a-cross-national-perspective-on-the-gendered-nature-of-couples-employment-preconditions-for-parenthood/>.

The real study used restricted registers. Each country team built couple-year
files internally and shared only collapsed count tables. This project
simulates all three data levels, so each one can be inspected.

## First, a summary of the study

### question, variables

This article studies examines ACROSS COUNTRIES, a couple's probability of
conceiving a first child, given their employment-wage constellation, and given
relevant controls. So the main independent/explanatory variables of interest are:

- the couple's employment-wage constellation, which has 6 possible
  permutations/cateogires: Both jobless, male only works, female only works,
  both work male main earner, both work female main earner, both work equal
  earner (40-60% = equal earning)
- the country
- interactive term between country and employment constellation
- Controls include:
  - the household income (sum of wages), as this is en important precondition
    for having children
  - woman's age (group)
  - womans education
  - whether either partner is foreigner

### model, equation

The functional form of the model is originally a probability model of the
S-curve type: Pit(couple i conceiving at time t | given that they are hitherto
childless)

This probability function is then also the likelihood contribution of obs_it,
once you insert the characteristics of the obs (person-year) i_t, including
whether they conceived or not, into above functional form. Or more precisely,
the likelihood contribution is Pit^(yit) (1-Pit)^(1-yit), where yit=1 means
they conceived, and yit=0 means not.

The likelihood function then is the product of all observations' likelihood
contributions. We take log likelihood and maximize wrt to the parameter values
b, aka MLE as usual.

We then interpret the estimates in the TRANSFORMED probability function,
ln(Pit/(1-Pit)), because this transformation makes the dependent variable
LINEAR in terms of the parameters b, such that we can interpret the estimated
bs in constant ceteris paribus terms. ln(Pit/(1-pit)) = logit funciton = b0 +
b1x1 + b2x2 +.... s.t. dln(Pit/(1-pit))/dx1 = b1. Now when transforming the
probability with the logit function, the interpretation of bs becomes: p/(1-p)
is by definition the odds of something happening. The odds of something
happening (in this case the conceiving of first child) are defined as the
probability of the event happening versus the probability of it not happening.
So the interpretation of b1 is: When x1 increases by 1, the log ODDS of
conceiving a child increases by b1.

The article then reports the resulting estimaes in Table A2, in the form of
odds RATIOS, rather than just pure odds, because the odds are different for
every combination of characteristics, whereas the odds ratio stays constant.

Exponentiating the logit model gives

$$
\text{odds}_i = \frac{p_i}{1-p_i} = e^{b_0 + b_1x_{1i} + b_2x_{2i} + \dots} = e^{b_0}\times e^{b_1x_{1i}}\times e^{b_2x_{2i}}\times\dots
$$

So each couple-year's odds depend on all its characteristics at once. With four
age groups, five countries, six constellations, three income terciles, four
education levels and two origin categories, there are 2,880 combinations, each
with its own odds. Reporting "the odds" would mean a table with 2,880 cells.

The odds ratio for one variable cancels everything else out. Comparing jobless
couples with equal earners, all other characteristics the same:

$$
\frac{\text{odds}(\text{jobless})}{\text{odds}(\text{equal})} = \frac{e^{b_0}\cdots e^{b_{\text{jobless}}}\cdots}{e^{b_0}\cdots 1 \cdots} = e^{b_{\text{jobless}}}
$$

Every other factor appears identically in the numerator and the denominator, so
it drops out. The result, $e^{b_{\text{jobless}}}$, is the same whatever the
couple's age, education or income. One number summarises the effect for
everyone, which is the "constant ceteris-paribus effect" from earlier.

> **Sidenote:** The table does contain odds, in one place. The constant row is
> $e^{b_0}$: the odds of the reference couple-year. In Model 2 it is 0.133, for
> a Norwegian equal-earner couple with a woman aged 18–25, low education, low
> household income and native-born partners. That corresponds to a probability
> of 0.133 / 1.133 ≈ 0.117. From there, you can build the odds of any
> combination by multiplying the relevant odds ratios. For the same couple type
> but Danish and jobless:
>
> $$
> 0.133 \times \underbrace{0.581}_{\text{Denmark}} \times \underbrace{0.670}_{\text{jobless}} \times \underbrace{1.132}_{\text{DK}\times\text{jobless}} \approx 0.059
> $$
>
> That is a probability of about 0.059 / 1.059 ≈ 0.055.
>
> So the odds ratios are the building blocks: the full set of odds can be
> reconstructed from them. Probabilities don't factor like this. The
> probability-scale effect of being jobless differs across combinations,
> because of the S-curve. That's why the paper summarises probabilities with
> AMPPs and AMEs, as averages, rather than as a single coefficient.
>
> To get the probalities from the odds ratios: Multiply the odds ratios that
> apply to the couple type to get its odds, then convert odds to a probability.
>
> Household resources and education carry much of what Model 1 attributed to
> the constellation.
>
> $$
> \text{odds} = \text{constant} \times \text{OR}_{\text{Denmark}} \times \text{OR}_{\text{age}} \times \text{OR}_{\text{constellation}} \times \text{OR}_{\text{DK}\times\text{constellation}}\ (\times \text{controls in Model 2}), \qquad p = \frac{\text{odds}}{1+\text{odds}}
> $$
>
> For example, a Danish jobless couple with the woman aged 18–25 in Model 1 has
> odds of 0.187 × 0.835 × 0.358 × 0.952 = 0.053, so p = 0.053 / 1.053 ≈ 0.051.

### results, table A2

#### 1. Baseline: the Danish reference couple

The constant × the Denmark main effect gives the odds for a Danish equal-earner
couple with the woman aged 18–25:

- **Model 1:** 0.187 × 0.835 = 0.156 odds, *p* ≈ 0.135.
- **Model 2:** 0.133 × 0.581 = 0.077 odds, *p* ≈ 0.072. In Model 2 this
  reference couple also has low education, low household wages and native-born
  partners.

Compared with Norway, Danish equal earners have 17% lower odds in Model 1 and
42% lower odds in Model 2. The bigger Model 2 gap arises because the reference
is now a low-income couple, and income matters more in Denmark.

#### 2. Couple constellation (reference: dual earners, equal wages)

| Constellation            | Model 1 OR                | Odds vs equal earners | Model 2 OR                | Odds vs equal earners |
|--------------------------|---------------------------|----------------------:|---------------------------|----------------------:|
| Jobless couple           | 0.358 × 0.952 = **0.341** |                  −66% | 0.670 × 1.132 = **0.758** |                  −24% |
| Male breadwinner         | 0.566 × 1.015 = **0.574** |                  −43% | 0.951 × 0.936 = **0.890** |                  −11% |
| Female breadwinner       | 0.506 × 0.982 = **0.497** |                  −50% | 0.799 × 1.041 = **0.832** |                  −17% |
| Dual, male main earner   | 0.800 × 0.854 = **0.683** |                  −32% | 0.864 × 0.919 = **0.794** |                  −21% |
| Dual, female main earner | 0.799 × 0.935 = **0.747** |                  −25% | 0.921 × 1.107 = **1.020** |                   +2% |

**Reading it:**

- **Every constellation has lower odds than equal earners**, except female main
  earners in Model 2, who are on a par with them.
- **The controls shrink every gap.** The jobless-couple disadvantage falls from
  −66% to −24%. Much of the gross gap reflects education and income differences
  between couple types.
- **Compared with Norway (the interaction p-values):**
  - In Model 1, the male-breadwinner (p = 0.22) and female-breadwinner
    (p = 0.29) effects are not significantly different from Norway's. The
    jobless-couple effect (p = 0.03) and both main-earner effects (p < 0.001)
    are.
  - In Model 2, all five Danish interactions differ significantly from Norway
    (p ≤ 0.034).
  - The male-main-earner disadvantage is consistently **larger** in Denmark than
    in Norway: its interactions are below 1 in both models.
  - The female-main-earner disadvantage is **smaller** in Denmark in Model 2
    (interaction 1.107).

#### 3. The gender contrasts

| Contrast                                | Model 1                  | Model 2                  |
|-----------------------------------------|--------------------------|--------------------------|
| **Employment:** male BW / female BW     | 0.574 / 0.497 = **1.16** | 0.890 / 0.832 = **1.07** |
| **Wages:** female ME / male ME          | 0.747 / 0.683 = **1.09** | 1.020 / 0.794 = **1.28** |

- **Employment:** male-breadwinner couples have 16% higher odds than
  female-breadwinner couples in Model 1, and 7% higher in Model 2. This is the
  traditional direction (H2), though it weakens with controls.
- **Wages:** female-main-earner couples have 9% higher odds than
  male-main-earner couples, rising to **28%** after controls. This is the
  non-traditional result, against both H1 and H2, and it sharpens once
  education and income are held equal.

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
   - *Generated from the paper's model:* whether a childless woman with a
     partner conceives that year. The probability comes straight from the
     paper's logit equation (eq. 1) with added controls (Model 2) and the
     Danish estimates in its Table A2: the couple's employment-wage
     constellation, her age and education, household wages and foreign origin.

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
python run_all.py          # stages 1-3, about 20 seconds
python run_all.py 2 3      # only the later stages
python run_all.py 0        # download the Statistics Denmark figures again (stage 0)
```

## Where things are

| File | Contents |
|---|---|
| `stage0_data.py` | downloads the Statistics Denmark figures into `0data/`, with a comment naming the table behind each download |
| `stage1_simulate_registers.py` | the one assumption and the simulation |
| `stage2_process_data.py` | who is in the sample, the derived variables, and the count table |
| `stage3_analyse.py` | the model, the comparison of odds ratios with Table A2 as a table and as charts, and the odds for each constellation |
| `common.py` | file locations, the ages at risk, category lists, the classification rules, and reading the paper's Danish odds ratios |
| `0data/` | the three tables of Statistics Denmark figures that stage 1 draws from, and `paper_table_a2_model2_denmark.csv`, the paper's odds ratios for Denmark |
| `1simulated_register_data/` | the five simulated registers: `bef`, `ras`, `ind`, `udda`, `fod` |
| `2processed_data/` | the couple-year file and the count table |
| `3estimation/` | the estimated odds ratios next to the paper's Danish ones, as a table and as charts by constellation, household wages and age; and the odds themselves for each constellation, written as fractions such as "1 to 13" |

## Where the numbers come from

- **First births:** the paper's logit equation with the Danish Model 2 odds
  ratios from Table A2. This is the only part that must match the paper.
  Table A2 uses Norway as the reference country, so it prints no Danish odds
  ratio directly: each one is the product of a main row and a "Denmark * …"
  row (for jobless couples, 0.670 × 1.132 = 0.758). The Danish values were
  worked out once and are kept as a ready-made input,
  `0data/paper_table_a2_model2_denmark.csv`, which stages 1 and 3 read.
- **Everything else:** Statistics Denmark's StatBank (statistikbanken.dk),
  downloaded by stage 0 through the public API into three small tables in
  `0data/`.

  | Trait | StatBank table | Year | File in `0data/` |
  |---|---|---|---|
  | How many childless women, per birth year | BARNLOS, FOLK1E | 2010 | `women_2010.csv` |
  | Living with a partner, by age | FAM100N | 2010 | `women_2010.csv` |
  | Born abroad | FOLK1E | 2010 | `immigrant_share_2010.csv` |
  | Partner's age | FOD11, VIE825 | 2010 | typed into stage 1, with its source |
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

## Results

- **The estimates reproduce Table A2.** The paper's Danish odds ratio lies inside
  the 95% confidence interval for 14 of the 15 estimates. The exception is
  high education (estimated 1.33, interval 1.30–1.366, paper 1.366), just
  outside: with 15 intervals at 95%, one falling outside by chance is expected.
  For the constellations, relative to dual equal earners:

  | Constellation | Estimated | Table A2 |
  |---|---|---|
  | Both jobless | 0.76 | 0.76 |
  | Only he works | 0.90 | 0.89 |
  | Only she works | 0.86 | 0.83 |
  | Both work, he earns more | 0.80 | 0.79 |
  | Both work, she earns more | 1.04 | 1.02 |

- **Size.** With 20% of Denmark's childless women: 152,000 women,
  78,000 partners, 366,000 couple-years and 39,000 first births.

## Interpreting results

**Odds, intuitively.** Odds compare how often something happens with how often it doesn't. Take 1,000 Danish reference couples in one year: about 72 conceive and 928 don't. Their odds are 72 to 928, or roughly 1 to 13: for every couple that conceives, about 13 don't.

**The odds ratio** then says how that "1 to 13" shifts for another group. For jobless couples it is 0.76. Multiply the first number: 0.76 to 13, or about 1 to 17. So where equal-earner couples have one conception for every 13 without, jobless couples have one for every 17.

In this study the event is rare enough (under about 20% a year) that you can read an odds ratio as almost the same thing as a ratio of probabilities. So as a working rule here:

| Odds ratio | Read it as |
|---|---|
| 0.76 | roughly 24% fewer conceptions than equal earners |
| 0.90 | roughly 10% fewer |
| 1.04 | about the same |

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
