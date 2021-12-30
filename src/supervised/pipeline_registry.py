# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from kedro.extras.datasets import text, yaml, pickle, pandas

from .pipelines.data import mushroom, mnist, kddcup99, adult
from .pipelines.algorithms import create_pipelines


def register_pipelines(**kwargs) -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    m_data = mushroom.create_data_pipeline()
    m_algs = create_pipelines(
        data_name="mushroom",
        kedro_train_x="mushroom_train_x",
        kedro_train_y="mushroom_train_y",
        kedro_test_x="mushroom_test_x",
        kedro_test_y="mushroom_test_y",
    )
    all_m = m_data + m_algs['mushroom_all_algs']
    
    mnist_data = mnist.create_data_pipeline()
    mnist_algs = create_pipelines(
        data_name="mnist",
        kedro_train_x="mnist_train_x",
        kedro_train_y="mnist_train_y",
        kedro_test_x="mnist_test_x",
        kedro_test_y="mnist_test_y",
    )
    # TODO remove "all_algs" and just sum them
    all_mnist = mnist_data + mnist_algs['mnist_all_algs']

    mnist_sub_data = mnist.create_sub_data_pipeline()
    mnist_sub_algs = create_pipelines(
        data_name="mnist_sub",
        kedro_train_x="mnist_sub_train_x",
        kedro_train_y="mnist_sub_train_y",
        kedro_test_x="mnist_sub_test_x",
        kedro_test_y="mnist_sub_test_y",
    )
    all_mnist_sub = mnist_sub_data + mnist_sub_algs['mnist_sub_all_algs']

    kddcup99_data = kddcup99.create_data_pipeline()
    kddcup99_algs = create_pipelines(
        data_name="kddcup99",
        kedro_train_x="kddcup99_train_x",
        kedro_train_y="kddcup99_train_y",
        kedro_test_x="kddcup99_test_x",
        kedro_test_y="kddcup99_test_y",
    )
    all_kddcup99 = kddcup99_data + kddcup99_algs['kddcup99_all_algs']

    adult_data = adult.create_data_pipeline()
    adult_algs = create_pipelines(
        data_name="adult",
        kedro_train_x="adult_train_x",
        kedro_train_y="adult_train_y",
        kedro_test_x="adult_test_x",
        kedro_test_y="adult_test_y",
    )
    all_adult = adult_data + adult_algs['adult_all_algs']
    
    return {
        **m_algs, 
        "mushroom_data": m_data,
        "mushroom": m_data + m_algs['mushroom_all_algs'],

        **mnist_algs,
        "mnist_data": mnist_data,
        "mnist": all_mnist,

        **mnist_sub_algs,
        "mnist_sub_data": mnist_sub_data,
        "mnist_sub": all_mnist_sub,

        **kddcup99_algs,
        "kddcup99_data": kddcup99_data,
        "kddcup99": all_kddcup99,

        **adult_algs,
        "adult_data": adult_data,
        "adult": all_adult,

        "goal": all_mnist_sub + all_adult,
        "__default__": all_mnist + all_kddcup99 + all_adult + all_m + all_mnist_sub,
        #"__default__": m_data + m_algs['mushroom_all_algs'] + mnist_data + mnist_algs['mnist_all_algs'],
    }
