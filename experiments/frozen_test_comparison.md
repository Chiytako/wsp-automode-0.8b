# Frozen Test Single Evaluation Comparison (`frozen_test`, n=200)

> Executed once under SHA-256 verification. Weights were finalized prior to unlocking and never re-tuned afterward.

| System | `% modified` (`||ΔW||_F`) | Proxy | `real_cf` AUROC (SR) | `real_cf` Recall / FPR | `real_cf` Flip | `consent_5p` Acc (`P2 / P3`) | `P5a / P5b` (`Flip`) | `long_50t` Acc (`recap ON`) | `decider_replay` Acc / ECE |
|---|---:|---:|---|---|---|---|---|---|---|
| `frozen_track0a_zeroshot` | `0.0%` (`0.0`) | `-1.2539` | `0.6587` (`0.6353`) | `0.025 / 0.0` | `0.9875` | `0.3958` (`0.875 / 0.125`) | `1.0 / 0.0` (`0.8958`) | `0.5833` (`0.6667`) | `0.5 / 0.1989` |
| `frozen_track0c_oracle_chat` | `0.0%` (`0.0`) | `-0.56906` | `0.6416` (`0.66`) | `1.0 / 0.9` | `0.275` | `0.5208` (`0.375 / 0.625`) | `0.625 / 0.5` (`0.2292`) | `0.5` (`0.5`) | `0.5 / 0.1989` |
| `frozen_track1_best` | `0.001633%` (`8.010948`) | `-0.02822` | `0.8213` (`0.8241`) | `0.85 / 0.475` | `0.2` | `0.6875` (`0.125 / 0.875`) | `0.375 / 0.75` (`0.2708`) | `0.5833` (`0.5833`) | `0.5 / 0.2138` |
| `frozen_track2_best` | `0.003264%` (`11.888532`) | `0.22845` | `0.8347` (`0.8391`) | `0.8 / 0.35` | `0.2375` | `0.7292` (`0.375 / 0.875`) | `0.5 / 0.625` (`0.1667`) | `0.5833` (`0.5833`) | `0.5 / 0.2135` |

## Detailed `decider_replay` & `second_reading` Breakdown

- **`frozen_track0a_zeroshot`**: `real_cf` ECE=`0.1728`, MarginSep=`0.1625`, SR Recall/FPR=`0.75/0.475`; `consent_5p` P3-P2=`0.0625`, P5b-P5a=`0.1406`, SR Acc=`0.6667`; `long_50t` EarlyAllow=`1.0`, MismatchBlock=`0.1667`; `decider_replay` by type=`{'noul': 0.6, 'choice': 0.52, 'score': 0.3333}`
- **`frozen_track0c_oracle_chat`**: `real_cf` ECE=`0.0959`, MarginSep=`0.1594`, SR Recall/FPR=`0.925/0.8`; `consent_5p` P3-P2=`0.0`, P5b-P5a=`0.0312`, SR Acc=`0.4583`; `long_50t` EarlyAllow=`0.0`, MismatchBlock=`1.0`; `decider_replay` by type=`{'noul': 0.6, 'choice': 0.52, 'score': 0.3333}`
- **`frozen_track1_best`**: `real_cf` ECE=`0.1424`, MarginSep=`2.225`, SR Recall/FPR=`0.825/0.475`; `consent_5p` P3-P2=`0.4844`, P5b-P5a=`0.5781`, SR Acc=`0.7292`; `long_50t` EarlyAllow=`0.5`, MismatchBlock=`0.6667`; `decider_replay` by type=`{'noul': 0.6, 'choice': 0.52, 'score': 0.3333}`
- **`frozen_track2_best`**: `real_cf` ECE=`0.106`, MarginSep=`2.3344`, SR Recall/FPR=`0.85/0.375`; `consent_5p` P3-P2=`0.6719`, P5b-P5a=`0.625`, SR Acc=`0.7083`; `long_50t` EarlyAllow=`0.8333`, MismatchBlock=`0.3333`; `decider_replay` by type=`{'noul': 0.6, 'choice': 0.52, 'score': 0.3333}`
