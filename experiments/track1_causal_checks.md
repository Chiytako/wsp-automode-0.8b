# Track 1 3-Point Causal Sanity Checks

- **`step1a_c2w_mlp_L15`** (`step1a_c2w`): **Kept=`False`** | Causal=`False` | ReplayOK=`True` | `best_scale=1.25` | `base=-1.20515` -> `best=-0.90032` | Spearman=`0.6` | Rev(-1)=`-1.46826` | RandMaxGain=`0.07262` vs CandGain=`0.30483`
  - Fail reasons: `['Dose-Response failed (gain=0.30483, spearman=0.600)']`
- **`step1a_c2w_attn_L15`** (`step1a_c2w`): **Kept=`False`** | Causal=`False` | ReplayOK=`True` | `best_scale=1.25` | `base=-1.20515` -> `best=-1.00314` | Spearman=`0.6` | Rev(-1)=`-1.34434` | RandMaxGain=`0.03864` vs CandGain=`0.20201`
  - Fail reasons: `['Dose-Response failed (gain=0.20201, spearman=0.600)']`
- **`step1a_c2w_mlp_L20`** (`step1a_c2w`): **Kept=`False`** | Causal=`False` | ReplayOK=`True` | `best_scale=1.0` | `base=-1.20515` -> `best=-1.15037` | Spearman=`1.0` | Rev(-1)=`-1.29854` | RandMaxGain=`0.14117` vs CandGain=`0.05478`
  - Fail reasons: `['Orthogonal Control failed (cand_gain=0.05478, rand_max=0.14117, rand_mean=0.04648, rand_std=0.07423)']`
- **`step1b_v_consent_mlp_L19`** (`step1b_consent`): **Kept=`False`** | Causal=`False` | ReplayOK=`True` | `best_scale=1.0` | `base=-0.31345` -> `best=-0.14915` | Spearman=`0.7` | Rev(-1)=`-0.3441` | RandMaxGain=`0.16395` vs CandGain=`0.1643`
  - Fail reasons: `['Dose-Response failed (gain=0.16430, spearman=0.700)', 'Orthogonal Control failed (cand_gain=0.16430, rand_max=0.16395, rand_mean=0.09810, rand_std=0.05313)']`
- **`step1b_v_consent_mlp_L15`** (`step1b_consent`): **Kept=`False`** | Causal=`False` | ReplayOK=`True` | `best_scale=0.25` | `base=-0.31345` -> `best=-0.1845` | Spearman=`0.1` | Rev(-1)=`-0.3118` | RandMaxGain=`0.1297` vs CandGain=`0.12895`
  - Fail reasons: `['Dose-Response failed (gain=0.12895, spearman=0.100)', 'Sign-Reversal failed (score(-1.0)=-0.31180 vs base=-0.31345)', 'Orthogonal Control failed (cand_gain=0.12895, rand_max=0.12970, rand_mean=0.08773, rand_std=0.03965)']`
- **`step1b_swiglu_order_gate_L15`** (`step1b_swiglu_gate`): **Kept=`True`** | Causal=`True` | ReplayOK=`True` | `best_scale=1.0` | `base=-1.20515` -> `best=0.22409` | Spearman=`1.0` | Rev(-1)=`-5.89519` | RandMaxGain=`0.41833` vs CandGain=`1.42924`
- **`step1b_swiglu_order_gate_L23_complement`** (`step1b_swiglu_gate`): **Kept=`False`** | Causal=`False` | ReplayOK=`True` | `best_scale=1.25` | `base=0.22409` -> `best=0.60579` | Spearman=`0.7` | Rev(-1)=`-6.44017` | RandMaxGain=`-1.17333` vs CandGain=`0.3817`
  - Fail reasons: `['Dose-Response failed (gain=0.38170, spearman=0.700)']`
