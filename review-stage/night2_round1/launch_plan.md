# Launch plan once timing test finishes

## Decision tree based on per-rule cost

| per-rule | scope | total expected | command |
|---|---|---|---|
| < 30s | all 55 rules | ~18 min + 55×30s = 46 min | `--rules ALL` |
| 30-90s | all 55 rules | ~18 min + 55×60s = 73 min | `--rules ALL` |
| 90-180s | 20 representative rules | ~18 min + 20×120s = 58 min | `--rules <subset>` |
| > 180s | 10 rules | ~18 min + 10×180s = 48 min | `--rules <subset>` |

(There are 61 rules in GALLERY_RULES and 60 in frozen_exemplars, intersection 55.)

Pool build + mixed-class ID + split is amortized once at ~18 min total.

## 20-rule representative subset (verified to exist in both GALLERY_RULES and exemplars)
```
all_red, all_same_suit, all_same_color, all_even, all_odd,
all_but_one_same_color, all_clubs_or_hearts,
straight5, straight5_same_color, strict_increasing,
ap_step1_len3_adj, ap_step2_len4_adj_ordered,
triple_2s_pos234, triple_any_adjacent, triple_any_pos345,
pair_jacks_pos45, four_of_a_kind_adjacent,
two_pairs_suits, two_pairs_ranks,
adjacent_share_rank_or_suit
```

## 10-rule subset (smaller fallback, verified)
```
all_red, all_even, all_but_one_same_color,
straight5, strict_increasing, triple_2s_pos234,
triple_any_adjacent, four_of_a_kind_adjacent,
two_pairs_suits, adjacent_share_rank_or_suit
```

## Launch command (full 55 rules)
```bash
nohup caffeinate -d -i -s ~/miniforge3/bin/python \
  review-stage/experiments/night2/full_sensitivity_audit.py \
  --depth 7 --max-programs 300000 --rules ALL \
  --output-dir review-stage/experiments/night2 \
  > review-stage/experiments/night2/sensitivity.log 2>&1 &
echo $! > review-stage/experiments/night2/sensitivity.pid
```

## Memory plan
- Kill timing test process before launching full audit (free RSS).
- Wait until full audit RSS stabilizes after merge step.
- Then launch adversarial driver in foreground.

## Adversarial driver launch (after audit reaches per-rule loop)
```bash
nohup caffeinate -d -i -s ~/miniforge3/bin/python \
  review-stage/experiments/night2/run_adversarial_hands.py \
  --depth 7 --max-programs 300000 \
  --n-candidates 50000 --top-k-diagnostic 20 --top-k-adversarial 10 \
  > review-stage/experiments/night2/adversarial.log 2>&1 &
echo $! > review-stage/experiments/night2/adversarial.pid
```

Expected runtime: ~12 min pool build + 5 rules × ~60s = ~17 min.
