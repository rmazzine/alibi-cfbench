import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs

import sys

sys.path.append('../')

import time

import numpy as np
import pandas as pd

from cfbench.cfbench import BenchmarkCF, TOTAL_FACTUAL
from benchmark.utils import timeout, TimeoutError

from alibi.explainers import CounterfactualProto


# Get initial and final index if provided
if len(sys.argv) == 3:
    initial_idx = sys.argv[1]
    final_idx = sys.argv[2]
else:
    initial_idx = 0
    final_idx = TOTAL_FACTUAL

# Create Benchmark Generator
benchmark_generator = BenchmarkCF(
    output_number=2,
    show_progress=True,
    disable_tf2=True,
    disable_gpu=True,
    initial_idx=int(initial_idx),
    final_idx=int(final_idx)).create_generator()


# The Benchmark loop
alibi_current_dataset = None
for benchmark_data in benchmark_generator:

    # Get Keras TensorFlow model
    original_model = benchmark_data['model']

    # Get train data
    non_ohe_columns = list(benchmark_data['df_train'].columns)[:-1]
    train_data = benchmark_data['df_oh_train']
    train_data_x = train_data.drop(columns=['output'])
    shape = (1, train_data_x.shape[1])

    # Get columns info
    columns = list(train_data.columns)[:-1]
    columns_prefix = [col.split('_')[0] for col in columns]
    columns_prefix_int = [int(c) for c in columns_prefix]
    cat_feats = benchmark_data['cat_feats']
    cat_feats_int = [int(c) for c in cat_feats]
    count_cat_prefix = [columns_prefix_int.count(cat) for cat in cat_feats_int]

    # https://github.com/SeldonIO/alibi/issues/366 => Remove binary features (treat as numerical)
    # since it is not supported
    # Get OHE parameters
    cat_vars = {}
    for cat_prefix, cat_count in zip(cat_feats_int, count_cat_prefix):
        if cat_count > 1:
            cat_vars[columns_prefix_int.index(cat_prefix)] = cat_count

    # Get feature ranges
    feat_range_down = [-1 if c in cat_feats else train_data.loc[:, c].min() for c in non_ohe_columns]
    feat_range_up = [1 if c in cat_feats else train_data.loc[:, c].max() for c in non_ohe_columns]

    # Get factual row as pd.Series
    factual_row = pd.Series(benchmark_data['factual_oh'], index=columns)

    # Get Evaluator
    evaluator = benchmark_data['cf_evaluator']

    if alibi_current_dataset != benchmark_data['dsname']:

        # Generator using gradients
        cf_generator = CounterfactualProto(
            original_model,
            shape,
            feature_range=(
                np.array([feat_range_down]).astype(np.float32),
                np.array([feat_range_up]).astype(np.float32)),
            cat_vars=cat_vars,
            ohe=True if len(cat_vars) > 0 else False
        )

        # Generator not using gradients
        cf_nograd_generator = CounterfactualProto(
            lambda x: original_model.predict(x),
            shape,
            feature_range=(
                np.array([feat_range_down]).astype(np.float32),
                np.array([feat_range_up]).astype(np.float32)),
            cat_vars=cat_vars,
            ohe=True if len(cat_vars) > 0 else False
        )

        cf_generator.fit(train_data_x.to_numpy())
        cf_nograd_generator.fit(train_data_x.to_numpy())

        # Update current dataset being experimented
        alibi_current_dataset = benchmark_data['dsname']


    @timeout(600)
    def generate_cf(cf_generator, cf_generator_name):
        try:
            # Create CF using ALIBI's explainer and measure generation time
            start_generation_time = time.time()
            explanation = cf_generator.explain(np.array([factual_row.to_list()]))
            cf_generation_time = time.time() - start_generation_time

            # Get first CF
            cf = explanation.cf['X'][0].tolist()
            if cf is None:
                cf = factual_row.to_list()
        except:
            # In case the CF generation fails, return same as factual
            cf = factual_row.to_list()
            cf_generation_time = np.NaN

        # Evaluate CF
        evaluator(
            cf_out=cf,
            algorithm_name=cf_generator_name,
            cf_generation_time=cf_generation_time,
            save_results=True)

    try:
        generate_cf(cf_generator, 'alibi')
    except TimeoutError:
        # If CF generation time exceeded the limit
        evaluator(
            cf_out=factual_row.to_list(),
            algorithm_name='alibi',
            cf_generation_time=np.NaN,
            save_results=True)

    try:
        generate_cf(cf_nograd_generator, 'alibi_nograd')
    except TimeoutError:
        # If CF generation time exceeded the limit
        evaluator(
            cf_out=factual_row.to_list(),
            algorithm_name='alibi_nograd',
            cf_generation_time=np.NaN,
            save_results=True)
