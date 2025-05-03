# How to get ulira scores

## One time set up:
0. create training masks
1. Train ulira oracles

## Testing a particular forget set and method

0. create forget masks
    * `unlearning/auditors/make_ulira_forget_masks.ipynb`
1. Train unlearnings according to forget_masks
    * `unlearning/auditors/run_ulira_unlearnings.py`
2. Compute ULIRA score
    * `unlearning/auditors/compute_ulira_scores.ipynb`
