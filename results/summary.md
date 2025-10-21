# Results

Spearman correlation, out-of-fold with position-held-out 5-fold CV, mean over 3 split seeds, across 17 ProteinGym assays.

## Average over assays

| model | mean Spearman | median |
|---|---|---|
| zeroshot | 0.275 | 0.255 |
| ridge | 0.446 | 0.430 |
| gnn-identity | 0.387 | 0.400 |
| gnn-window | 0.395 | 0.396 |
| gnn-contact | 0.411 | 0.403 |
| gnn-identity+llr | 0.424 | 0.438 |
| gnn-window+llr | 0.433 | 0.432 |
| gnn-contact+llr | 0.445 | 0.431 |

## By selection type (mean Spearman)

| selection type | n assays | zeroshot | ridge | gnn-identity | gnn-window | gnn-contact | gnn-identity+llr | gnn-window+llr | gnn-contact+llr |
|---|---|---|---|---|---|---|---|---|---|
| Activity | 5 | 0.241 | 0.336 | 0.298 | 0.312 | 0.321 | 0.332 | 0.347 | 0.352 |
| Binding | 1 | 0.529 | 0.585 | 0.501 | 0.547 | 0.566 | 0.591 | 0.613 | 0.632 |
| Expression | 2 | 0.279 | 0.394 | 0.371 | 0.384 | 0.384 | 0.384 | 0.381 | 0.377 |
| OrganismalFitness | 4 | 0.241 | 0.412 | 0.415 | 0.430 | 0.454 | 0.439 | 0.454 | 0.477 |
| Stability | 5 | 0.284 | 0.578 | 0.438 | 0.422 | 0.446 | 0.486 | 0.489 | 0.502 |

## Does 3D structure help?

Paired differences across assays (first model minus second). The interval is a
95% bootstrap over assays; "wins" counts assays where the first model is better.

| comparison | mean diff | 95% CI | wins |
|---|---|---|---|
| contact vs identity (no zero-shot input) | +0.024 | [+0.010, +0.040] | 13/17 |
| contact vs sequence window (no zero-shot input) | +0.016 | [+0.008, +0.026] | 14/17 |
| contact vs identity (+ zero-shot input) | +0.021 | [+0.010, +0.034] | 14/17 |
| contact vs sequence window (+ zero-shot input) | +0.011 | [+0.002, +0.023] | 10/17 |
| zero-shot input added to the contact GNN | +0.034 | [+0.011, +0.059] | 12/17 |
| contact GNN + zero-shot vs ridge | -0.002 | [-0.033, +0.029] | 9/17 |

## Per assay

| assay | type | len | zeroshot | ridge | gnn-identity | gnn-window | gnn-contact | gnn-identity+llr | gnn-window+llr | gnn-contact+llr |
|---|---|---|---|---|---|---|---|---|---|---|
| A0A1I9GEU1_NEIME_Kennouche_2019 | Activity | 161 | -0.05 | 0.07 | 0.08 | 0.08 | 0.08 | 0.04 | 0.06 | 0.05 |
| AACC1_PSEAI_Dandage_2018 | OrganismalFitness | 177 | 0.25 | 0.36 | 0.40 | 0.40 | 0.40 | 0.44 | 0.43 | 0.43 |
| BLAT_ECOLX_Deng_2012 | OrganismalFitness | 286 | 0.32 | 0.32 | 0.37 | 0.39 | 0.39 | 0.40 | 0.42 | 0.42 |
| CASP7_HUMAN_Roychowdhury_2020 | Activity | 281 | 0.58 | 0.57 | 0.48 | 0.48 | 0.49 | 0.57 | 0.58 | 0.59 |
| CCDB_ECOLI_Tripathi_2016 | OrganismalFitness | 101 | -0.02 | 0.36 | 0.29 | 0.34 | 0.40 | 0.31 | 0.33 | 0.40 |
| DOCK1_MOUSE_Tsuboyama_2023_2M0Y | Stability | 66 | 0.54 | 0.65 | 0.52 | 0.48 | 0.51 | 0.58 | 0.56 | 0.57 |
| GFP_AEQVI_Sarkisyan_2016 | Activity | 238 | 0.07 | 0.25 | 0.19 | 0.23 | 0.25 | 0.21 | 0.25 | 0.25 |
| KCNE1_HUMAN_Muhammad_2023_expression | Expression | 129 | 0.16 | 0.36 | 0.27 | 0.29 | 0.28 | 0.28 | 0.26 | 0.25 |
| NUD15_HUMAN_Suiter_2020 | Expression | 164 | 0.40 | 0.43 | 0.47 | 0.48 | 0.49 | 0.49 | 0.50 | 0.50 |
| Q6WV12_9MAXI_Somermeyer_2022 | Activity | 222 | 0.08 | 0.19 | 0.22 | 0.23 | 0.24 | 0.21 | 0.22 | 0.24 |
| RASH_HUMAN_Bandaru_2017 | Activity | 189 | 0.53 | 0.60 | 0.52 | 0.55 | 0.54 | 0.63 | 0.63 | 0.64 |
| RASK_HUMAN_Weng_2022_binding-DARPin_K55 | Binding | 188 | 0.53 | 0.59 | 0.50 | 0.55 | 0.57 | 0.59 | 0.61 | 0.63 |
| RCD1_ARATH_Tsuboyama_2023_5OAO | Stability | 57 | 0.24 | 0.72 | 0.48 | 0.48 | 0.47 | 0.60 | 0.62 | 0.62 |
| RCRO_LAMBD_Tsuboyama_2023_1ORC | Stability | 63 | 0.26 | 0.41 | 0.33 | 0.28 | 0.32 | 0.35 | 0.31 | 0.37 |
| RL40A_YEAST_Roscoe_2013 | OrganismalFitness | 128 | 0.42 | 0.61 | 0.60 | 0.60 | 0.63 | 0.61 | 0.64 | 0.65 |
| SBI_STAAM_Tsuboyama_2023_2JVG | Stability | 56 | 0.25 | 0.45 | 0.32 | 0.36 | 0.39 | 0.33 | 0.35 | 0.35 |
| SPG2_STRSG_Tsuboyama_2023_5UBS | Stability | 56 | 0.12 | 0.66 | 0.54 | 0.52 | 0.54 | 0.58 | 0.60 | 0.60 |
