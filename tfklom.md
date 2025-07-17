# Teacher-forcing KLoM

## Overview

Teacher-forcing KLoM is an extension of the Knowledge Leakage of Margins (KLoM) metric designed specifically for language models. It evaluates margin divergence under teacher forcing conditions to assess knowledge retention in unlearned models.

## Key Concept

For a token sequence `x = (w₁, ..., wₜ)` and model `θ`, the method computes margin `φₜ(x; θ)` at each prediction step `t`. This margin represents the model's confidence in the true next token `wₜ₊₁` relative to alternative tokens, given the prefix `x<ₜ₊₁ = (w₁, ..., wₜ)`.

## Computation Process

### 1. Margin Calculation
- Uses logit-gap definition analogous to standard KLoM
- Input context corresponds to prefix `x<ₜ₊₁`
- True label is the next token `wₜ₊₁`

### 2. Histogram Comparison
At each prediction step `t`:
- Generate margin sets from oracle ensemble: `{φₜ(x; θᵢᵒ)}ᴺᵢ₌₁`
- Generate margin sets from unlearned ensemble: `{φₜ(x; θᵢᶠ)}ᴺᵢ₌₁`
- Create histograms: `Histᵒₜ(x)` and `Histᶠₜ(x)`
- Compute KL divergence: `KLoMₜ(x) = D_KL(Histᵒₜ(x) ∥ Histᶠₜ(x))`

### 3. Aggregation
- **Per sequence**: `KLoM(x) = (1/T) ∑ᵀₜ₌₁ KLoMₜ(x)`
- **Per dataset**: `KLoM(D) = (1/|D|) ∑ₓ∈D KLoM(x)`

## Key Advantages

1. **Preserves robustness**: Maintains the original KLoM metric's robust properties
2. **No hyperparameter changes**: Fully compatible with existing KLoM configurations
3. **Teacher-forcing compatibility**: Works seamlessly with standard autoregressive evaluation
4. **Gaming resistance**: Significantly harder to exploit than multiple-choice formats
5. **Comprehensive assessment**: Compares full predictive distributions rather than checking specific outputs
6. **Subtle knowledge detection**: Captures nuanced forms of retained knowledge that simpler metrics might miss

## Technical Benefits

The method enables distributional comparison for next-token predictions in autoregressive models while comparing against ground truth model distributions, making it particularly effective at detecting sophisticated knowledge retention strategies.

## Resilient Implementation

We employ a robust two-step approach to ensure efficient and fault-tolerant computation of Teacher-forcing KLoM:

### Step 1: Margin Computation
- **Check for existing margins**: Verify if margins have been precomputed for the given dataset and model ensembles
- **Compute missing margins**: If margins are not available, compute `φₜ(x; θ)` for all sequences, time steps, and models in both oracle and unlearned ensembles
- **Persistent storage**: Cache computed margins to disk to avoid recomputation in subsequent runs
- **Validation**: Ensure margin completeness across all required sequences and models

### Step 2: Histogram Generation and KLoM Scoring
- **Load precomputed margins**: Retrieve margins from Step 1 (either newly computed or previously cached)
- **Generate histograms**: Create `Histᵒₜ(x)` and `Histᶠₜ(x)` from margin sets at each time step
- **Compute KL divergence**: Calculate `KLoMₜ(x)` scores using histogram pairs
- **Aggregate scores**: Average across time steps and sequences to produce final KLoM metrics

This modular approach provides several benefits:
- **Efficiency**: Avoids redundant margin computations across multiple evaluation runs
- **Robustness**: Allows recovery from partial computations if interrupted
- **Scalability**: Enables parallel processing of margin computation and histogram generation
- **Debugging**: Facilitates inspection of intermediate results for validation and troubleshooting
