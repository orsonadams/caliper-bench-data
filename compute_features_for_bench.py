# /bin/env

# Make a series of assumption:
# 	* [curl, tensorflow] is installed
# 	* Modelset endpoint is reachable

# TODO
# 	* Docstrings
# 	* Improve type hints
# 	* Use a better default for test features spec

import json
import tensorflow as tf
import subprocess
import copy
import argparse
import logging
from collections.abc import Iterable

logger = logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)

SUPPORTED_FEATURE_TYPES = set(("str", "float", "int"))


def add_test_features(
    test_features: dict[str, (any, str)], current_features: dict[str, tf.train.Features]
) -> dict[str, tf.train.Feature]:
    _copy_of_current_features_to_update = copy.deepcopy(current_features)

    def _create_new_features(v, T):
        new_tf_feature = None

        if T == "str":
            new_tf_feature = tf.train.Feature(float_list=tf.train.ByteList(value=[v]))
        elif T == "float":
            new_tf_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[v]))
        elif T == "int":
            new_tf_feature = tf.train.Feature(float_list=tf.train.Int64List(value=[v]))
        return new_tf_feature

    for name, (value, feature_type) in test_features.items():
        new_feature = _create_new_features(value, feature_type)
        if new_feature is not None:
            _copy_of_current_features_to_update[name] = new_feature

    return _copy_of_current_features_to_update


def get_model_metadata(modelset_endpoint, show_stderr=False):
    cmd = ["curl", modelset_endpoint]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    o, e = proc.communicate()
    if show_stderr:
        print(e, str(proc.returncode))
    return json.loads(o.decode("utf-8"))


def compute_required_features(
    features_metadata: dict[str, any], features_to_remove: set
) -> set:
    feats = features_metadata["metadata"]["signature_def"]["signature_def"][
        "serving_feature_names"
    ]["outputs"]
    return set(feats.keys()) - features_to_remove


def compute_required_example(
    example: tf.train.Example,
    required_features: set,
    test_features: dict[str, (any, str)],
) -> tf.train.Example:
    features = {
        k: v for k, v in example.features.feature.items() if k in required_features
    }
    if test_features:
        features = add_test_features(test_features, features)

    return tf.train.Example(features=tf.train.Features(feature=features))


def read_and_update_examples(
    filename: str, required_features: set, test_features: dict[str, (any, str)]
):
    raw_dataset = tf.data.TFRecordDataset(filename)
    for record in raw_dataset:
        example = tf.train.Example().FromString(record.numpy())
        yield compute_required_example(example, required_features, test_features)


def write_examples_as_tf_records(
    output_filename: str, examples: list[tf.train.Example]
) -> None:
    with tf.io.TFRecordWriter(output_filename) as writer:
        for example in examples:
            writer.write(example.SerializeToString())


def load_test_feature_spec(filename: str) -> dict[str, (any, str)]:
    if filename == "{}":
        logging.warning("No test features provided; Continuing without them")
        return {}
    return json.loads(open(filename).read())


def parse_test_feature_spec(spec):
    for name, values in spec["enrich"].items():
        v, T = values
        if T not in SUPPORTED_FEATURE_TYPES:
            raise TypeError(
                f"Invalid test feature spec. {T} not in SUPPORTED_FEATURE_TYPES: {SUPPORTED_FEATURE_TYPES}"
            )
    maybe_remove = spec["remove"]
    if not isinstance(maybe_remove, Iterable):
        raise TypeError(
            f"Invalid test feature spec. {maybe_remove} is not an Iterable but should be"
        )
    return spec


def main(args):
    test_features = load_test_feature_spec(args.test_feature_spec_file)
    test_features_parsed = parse_test_feature_spec(test_features)
    required_features_metadata = get_model_metadata(args.modelset_endpoint)
    features_to_remove = set(test_features["remove"])
    required_features = compute_required_features(
        required_features_metadata, features_to_remove
    )
    updated_examples = read_and_update_examples(
        args.record_input_file, required_features, test_features_parsed["enrich"]
    )

    write_examples_as_tf_records(args.record_output_file, updated_examples)
    logging.info(f"Done writing updated examples to {args.record_output_file}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelset-endpoint", type=str, dest="modelset_endpoint")
    parser.add_argument("--record-input-file", type=str, dest="record_input_file")
    parser.add_argument("--record-output-file", type=str, dest="record_output_file")

    # TODO use a better default value for file name?
    parser.add_argument(
        "--test-feature-spec", type=str, dest="test_feature_spec_file", default="{}"
    )

    args = parser.parse_args()
    main(args)
