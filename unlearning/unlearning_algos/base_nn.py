"""
READ BEFORE You ADD AN ALGO:

all functions should be of the form:

def unlearning_algo_name(
    model: torch.nn.Module,
    train_dataloader: ch.utils.data.DataLoader,
    forget_indices: List[int],
    forget_dataloader: ch.utils.data.Subset,
    **kwargs,
) -> ch.nn.Module:

Pass in the model, together with either:
- the train dataloader and the indices to forget
- the forget dataloader

The function should return a new model with unlearning applied, and have
no side effects. In particular, the original model should not be modified.
"""

from unlearning.unlearning_algos.oracle_matching import oracle_matching, dm_matching
from unlearning.unlearning_algos.dummies import do_nothing, load_an_oracle, retrain_an_oracle
from unlearning.unlearning_algos.grad_ascent import (
    gradient_ascent,
    gradient_ascent_grid_search,
)
from unlearning.unlearning_algos.dm_direct import dm_direct

try:
    from unlearning.unlearning_benchmarks.benchmarks import (
        gradient_descent_wrapper as benchmark_GD_wrapper, )
    from unlearning.unlearning_benchmarks.benchmarks import (
        gradient_ascent_wrapper as benchmark_GA_wrapper, )
except ImportError as e:
    print("Tried importing unlearning_benchmarks, but failed")
    print(e)
    benchmark_GA_wrapper = None
    benchmark_GD_wrapper = None

from unlearning.unlearning_benchmarks.scrub import (scrub_wrapper as scrub)

NAME_TO_ALGO = {
    "do_nothing": do_nothing,
    "load_an_oracle": load_an_oracle,
    "retrain_an_oracle": retrain_an_oracle,
    "oracle_matching": oracle_matching,
    "gradient_ascent": gradient_ascent,
    "gradient_ascent_grid_search": gradient_ascent_grid_search,
    #
    "dm_direct": dm_direct,
    "dm_matching": dm_matching,
    #
    # Benchmark algorithms
    "benchmark_GD_wrapper": benchmark_GD_wrapper,
    "benchmark_GA_wrapper": benchmark_GA_wrapper,
    "scrub": scrub
    # "benchmark_CAN_wrapper": benchmark_CAN_wrapper,
}
