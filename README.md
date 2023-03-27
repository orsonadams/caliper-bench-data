Caliper Benchmarking Data ❚█══█❚

Build modelset specific benchmarking data. This script reads modelset metadata, grabs the features required to infer the model and filter (or adds) existing data to those features. Finally it writes the updated feature to disk which you can use later in you caliper benchmarking. Happy benching! 

Its invoked as follows:

```bash

python compute_features_for_bench.py 
--modelset-endpoint https://prod-ml-platform.etsycloud.com/barista/nr-third-pass-si/v1/models/nr-third-pass-si/metadata \
--record-input-file part-08996-45b5f53d-5dd4-4f05-8460-72b69d51798a-c000.tfrecord \
--record-output-file test-ouput.record \
--test-feature-spec test_feature_spec_lwr_bench.json

```

args:

* --modelset-endpoint: the metadata endpoint for the modelset.  
* --record-input-file: is the raw input data as a tfrecord file. These can be found in training logs here `gs://ml-systems-prod-attributed-mmx-logs-zjh13h/attributed_training_data/query_pipeline_web_organic/tfrecord/AttributedInstance/`

* --record-output-file: where should the script write the updates tfrecord

* --test-feature-spec: [OPTIONAL] a json file that looks like below. with a "remove" key of type `list[str]` which captures a list of feature names to remove. an "enrich" key which captures features that are not in the raw file that you'd like to include in updated tfrecord. Notice that you this of type dict[str, (value, type)]. Where `type` is the TF type with only python types `str`, `int`, `float` supported.

```json
{
    "enrich": {
        "candidateInfo.scores.score": [
            0.1,
            "float"
        ]
    },
    "remove": [
        "requestUUID"
    ]
}
```
