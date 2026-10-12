# STEP 8A — RULER within-sample ranking (8K, 125M)

Not long-range memory. 8K examples are in-SWA. Ghat is the K=32 JVP estimator.
G is not importance. A positive M2−M0 delta is not a router design.

## 1. Samples

- checkpoint: `125m_ttt_e2e_pretrain_dclm_8k_1x_cc`
- tasks: niah_single_2 / 8k and niah_multikey_2 / 8k, n=200 each
- split seed 20260909, 60/20/20 by sample_id

## 2–3. Candidate windows

```json
{
  "niah_single_2": {
    "kind": "single2",
    "n_samples": 200,
    "n_positive_windows": 200,
    "n_negative_windows": 49527,
    "kv_fully_inside_positive": 200,
    "kv_fully_inside_rate": 1.0,
    "positive_query_or_pad_overlap": 0,
    "placement_failed_positive": 0,
    "placement_failed_negative": 0,
    "negative_windows_per_sample": {
      "n": 200,
      "min": 230,
      "median": 249.0,
      "max": 249,
      "mean": 247.635
    },
    "problems": [],
    "stop": false
  },
  "niah_multikey_2": {
    "kind": "mk2",
    "n_samples": 200,
    "n_positive_windows": 200,
    "n_negative_windows": 88125,
    "kv_fully_inside_positive": 200,
    "kv_fully_inside_rate": 1.0,
    "positive_query_or_pad_overlap": 0,
    "placement_failed_positive": 0,
    "placement_failed_negative": 0,
    "negative_windows_per_sample": {
      "n": 200,
      "min": 422,
      "median": 447.0,
      "max": 447,
      "mean": 440.625
    },
    "problems": [],
    "stop": false
  }
}
```

## 4. Feature sanity

```json
{
  "niah_multikey_2": {
    "n_windows": 88325,
    "nll_finite": true,
    "ghat_finite": true,
    "ghat_nonneg": true,
    "mean_nll": 2.2029050365211,
    "mean_ghat": 5.656053682921663,
    "mean_log_ghat": -0.878784605128164,
    "spearman_window_nll_logg": 0.2977942943617069
  },
  "niah_single_2": {
    "n_windows": 49727,
    "nll_finite": true,
    "ghat_finite": true,
    "ghat_nonneg": true,
    "mean_nll": 3.283519310393339,
    "mean_ghat": 11.670448441698845,
    "mean_log_ghat": 2.121685275600273,
    "spearman_window_nll_logg": 0.5936137642838476
  }
}
```

## 5–7. Metrics, deltas, coefficients

```json
{
  "niah_single_2": {
    "validation": {
      "M0": {
        "mrr": 0.21245787402090457,
        "recall_at_1": 0.075,
        "recall_at_5": 0.375,
        "recall_at_1pct": 0.175,
        "normalized_rank": 0.03324867492410033,
        "percentile_rank": 0.9667513250758997,
        "rank": 9.175,
        "n_candidates": 247.45,
        "n_samples": 40.0
      },
      "M1": {
        "mrr": 0.12437133883463991,
        "recall_at_1": 0.05,
        "recall_at_5": 0.125,
        "recall_at_1pct": 0.075,
        "normalized_rank": 0.14549242680271002,
        "percentile_rank": 0.8545075731972901,
        "rank": 37.125,
        "n_candidates": 247.45,
        "n_samples": 40.0
      },
      "M2": {
        "mrr": 0.23963494372107816,
        "recall_at_1": 0.1,
        "recall_at_5": 0.375,
        "recall_at_1pct": 0.175,
        "normalized_rank": 0.03232267780800622,
        "percentile_rank": 0.9676773221919938,
        "rank": 8.9,
        "n_candidates": 247.45,
        "n_samples": 40.0
      },
      "M2_shuffled": {
        "mrr": 0.21329914968072866,
        "recall_at_1": 0.075,
        "recall_at_5": 0.375,
        "recall_at_1pct": 0.175,
        "normalized_rank": 0.03303957766550071,
        "percentile_rank": 0.9669604223344992,
        "rank": 9.125,
        "n_candidates": 247.45,
        "n_samples": 40.0
      }
    },
    "test": {
      "M0": {
        "mrr": 0.19231480067490128,
        "recall_at_1": 0.05,
        "recall_at_5": 0.25,
        "recall_at_1pct": 0.125,
        "normalized_rank": 0.030640591957454777,
        "percentile_rank": 0.9693594080425452,
        "rank": 8.575,
        "n_candidates": 248.475,
        "n_samples": 40.0
      },
      "M1": {
        "mrr": 0.12679977655295566,
        "recall_at_1": 0.025,
        "recall_at_5": 0.15,
        "recall_at_1pct": 0.125,
        "normalized_rank": 0.11614918467882569,
        "percentile_rank": 0.8838508153211743,
        "rank": 29.8,
        "n_candidates": 248.475,
        "n_samples": 40.0
      },
      "M2": {
        "mrr": 0.19867826617826617,
        "recall_at_1": 0.05,
        "recall_at_5": 0.35,
        "recall_at_1pct": 0.125,
        "normalized_rank": 0.029258702767939578,
        "percentile_rank": 0.9707412972320604,
        "rank": 8.225,
        "n_candidates": 248.475,
        "n_samples": 40.0
      },
      "M2_shuffled": {
        "mrr": 0.19396335024624495,
        "recall_at_1": 0.05,
        "recall_at_5": 0.275,
        "recall_at_1pct": 0.125,
        "normalized_rank": 0.03064888600320299,
        "percentile_rank": 0.9693511139967971,
        "rank": 8.575,
        "n_candidates": 248.475,
        "n_samples": 40.0
      }
    },
    "bootstrap_M2_minus_M0": {
      "mrr": {
        "point": 0.006363465503364885,
        "ci95_lo": -0.00046591898705868973,
        "ci95_hi": 0.0138641082053911,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      },
      "recall_at_1": {
        "point": 0.0,
        "ci95_lo": 0.0,
        "ci95_hi": 0.0,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      },
      "recall_at_5": {
        "point": 0.1,
        "ci95_lo": 0.025,
        "ci95_hi": 0.2,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      },
      "normalized_rank": {
        "point": -0.0013818891895152057,
        "ci95_lo": -0.004034555298067006,
        "ci95_hi": 0.0010030340495056098,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      }
    },
    "bootstrap_M2shuffled_minus_M0": {
      "mrr": {
        "point": 0.0016485495713436892,
        "ci95_lo": -0.0017429109901536375,
        "ci95_hi": 0.005592308987989133,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      },
      "recall_at_1": {
        "point": 0.0,
        "ci95_lo": 0.0,
        "ci95_hi": 0.0,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      },
      "recall_at_5": {
        "point": 0.025,
        "ci95_lo": 0.0,
        "ci95_hi": 0.075,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      },
      "normalized_rank": {
        "point": 8.294045748210264e-06,
        "ci95_lo": -0.0006945171992317093,
        "ci95_hi": 0.0007111052907281298,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      }
    },
    "coefficients": {
      "M0": [
        2.338448921218517
      ],
      "M1": [
        1.807261816222651
      ],
      "M2": [
        2.5690463395300918,
        -0.45203364851369654
      ],
      "M2_shuffled": [
        2.2613522893626135,
        0.03667046628915602
      ]
    },
    "lambda": {
      "M0": 0.0001,
      "M1": 0.0001,
      "M2": 0.001,
      "M2_shuffled": 0.001
    }
  },
  "niah_multikey_2": {
    "validation": {
      "M0": {
        "mrr": 0.009137621665763759,
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "recall_at_1pct": 0.0,
        "normalized_rank": 0.468915863524073,
        "percentile_rank": 0.5310841364759269,
        "rank": 208.175,
        "n_candidates": 443.0,
        "n_samples": 40.0
      },
      "M1": {
        "mrr": 0.009365473028552796,
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "recall_at_1pct": 0.0,
        "normalized_rank": 0.466544472364473,
        "percentile_rank": 0.5334555276355271,
        "rank": 207.075,
        "n_candidates": 443.0,
        "n_samples": 40.0
      },
      "M2": {
        "mrr": 0.009088665972979809,
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "recall_at_1pct": 0.0,
        "normalized_rank": 0.46827083134535663,
        "percentile_rank": 0.5317291686546435,
        "rank": 207.9,
        "n_candidates": 443.0,
        "n_samples": 40.0
      },
      "M2_shuffled": {
        "mrr": 0.006564173960435846,
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "recall_at_1pct": 0.0,
        "normalized_rank": 0.5125597718332856,
        "percentile_rank": 0.4874402281667144,
        "rank": 227.475,
        "n_candidates": 443.0,
        "n_samples": 40.0
      }
    },
    "test": {
      "M0": {
        "mrr": 0.011237682184519987,
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "recall_at_1pct": 0.0,
        "normalized_rank": 0.465054019953985,
        "percentile_rank": 0.534945980046015,
        "rank": 206.0,
        "n_candidates": 441.75,
        "n_samples": 40.0
      },
      "M1": {
        "mrr": 0.014217178307107722,
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "recall_at_1pct": 0.0,
        "normalized_rank": 0.44315552869578123,
        "percentile_rank": 0.5568444713042189,
        "rank": 195.85,
        "n_candidates": 441.75,
        "n_samples": 40.0
      },
      "M2": {
        "mrr": 0.011878563382382978,
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "recall_at_1pct": 0.0,
        "normalized_rank": 0.462487939607918,
        "percentile_rank": 0.537512060392082,
        "rank": 204.85,
        "n_candidates": 441.75,
        "n_samples": 40.0
      },
      "M2_shuffled": {
        "mrr": 0.011106933027900523,
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "recall_at_1pct": 0.0,
        "normalized_rank": 0.47499734936437754,
        "percentile_rank": 0.5250026506356225,
        "rank": 210.575,
        "n_candidates": 441.75,
        "n_samples": 40.0
      }
    },
    "bootstrap_M2_minus_M0": {
      "mrr": {
        "point": 0.0006408811978629904,
        "ci95_lo": -0.00021474548741751048,
        "ci95_hi": 0.0017784221131734266,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      },
      "recall_at_1": {
        "point": 0.0,
        "ci95_lo": 0.0,
        "ci95_hi": 0.0,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      },
      "recall_at_5": {
        "point": 0.0,
        "ci95_lo": 0.0,
        "ci95_hi": 0.0,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      },
      "normalized_rank": {
        "point": -0.002566080346066987,
        "ci95_lo": -0.009446789152538777,
        "ci95_hi": 0.004211697652597085,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      }
    },
    "bootstrap_M2shuffled_minus_M0": {
      "mrr": {
        "point": -0.0001307491566194644,
        "ci95_lo": -0.005086480971614039,
        "ci95_hi": 0.005729008426095015,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      },
      "recall_at_1": {
        "point": 0.0,
        "ci95_lo": 0.0,
        "ci95_hi": 0.0,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      },
      "recall_at_5": {
        "point": 0.0,
        "ci95_lo": 0.0,
        "ci95_hi": 0.0,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      },
      "normalized_rank": {
        "point": 0.009943329410392616,
        "ci95_lo": -0.03773812316443483,
        "ci95_hi": 0.054454273884877565,
        "n_samples": 40,
        "n_bootstrap": 10000,
        "unit": "sample"
      }
    },
    "coefficients": {
      "M0": [
        0.05235334406866291
      ],
      "M1": [
        0.004423540421007437
      ],
      "M2": [
        0.002181890822111882,
        0.00016957882174839707
      ],
      "M2_shuffled": [
        0.05148004490683853,
        0.03118742075480316
      ]
    },
    "lambda": {
      "M0": 0.0001,
      "M1": 0.0001,
      "M2": 10.0,
      "M2_shuffled": 0.0001
    }
  }
}
```

## 8–10. Leakage / interpretation

Allowed claim scope: within-sample ranking of L=32 token windows on in-SWA 8K RULER.
M2 vs M0 asks whether log(Ghat) adds ranking signal beyond mean NLL on the same window.
Shuffle control: M2-shuffled should be near M0. Improvement under shuffle is treated as fitting artifact, not a G signal.
Surprisal remains the first confounder (STEP 7A Spearman(Ghat,NLL)≈0.80 at token grain).
Do not treat a ranking gain as evidence for explicit memory, a router, or causal importance.
8K is inside SWA=8192; this is not a long-range TTT-memory result.

single_2: NLL already separates needle vs haystack (pos-neg Δmean NLL ≈ 1.46). log(G) is correlated with NLL at window grain (Spearman 0.59). M2 weight on standardized log(G) is negative. M2-shuffled ≈ M0. Pre-specified ΔMRR / ΔR@1 / Δnormalized-rank CIs include 0. ΔR@5 = +0.10 with CI (0.025, 0.20) excluding 0; do not promote this secondary metric over the pre-specified core. Largest test rank move is sid=196 (35→26).

MK2: relevant vs distractor windows are format-matched needles. pos-neg Δmean NLL ≈ 0.008. All scorers near chance (mean rank ≈ 200 / 442). M2 selected λ=10 and near-zero weights. Future-relevance is not linearly readable from NLL or G at needle time (query is still in the future).

Minimum allowed reading: this 8K ranking test does **not** support using Ghat as an extra memory-ranking signal beyond NLL. It does not falsify every possible G utility; it falsifies a stable linear within-sample ranking gain on these two splits.

