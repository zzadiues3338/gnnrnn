# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 11:44:41 2025

@author: 14055
"""

image_uri = retrieve("xgboost", boto3.Session().region_name, "0.90-1")

model = Model(
    image_uri=image_uri,
    model_data=model_url,
    role=role,
    sagemaker_session=pipeline_session,
)

create_model_step = ModelStep(
    name="CreateXGBoostModelStep",
    step_args=model.create(),
)

transformer = Transformer(
    model_name=create_model_step.properties.ModelName,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    accept="text/csv",
    assemble_with="Line",
    output_path=transform_output_path,
    sagemaker_session=pipeline_session,
)

transform_input_param = ParameterString(
    name="transform_input",
    default_value=f"s3://{bucket}/{prefix}/transform_input/validation",
)

transform_arg = transformer.transform(
    transform_input_param,
    content_type="text/csv",
    split_type="Line",
    # exclude the ground truth (first column) from the validation set
    # when doing inference.
    input_filter="$[1:]",
)

from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.quality_check_step import DataQualityCheckConfig


baseline_prefix = prefix + "/baselining"
baseline_data_prefix = baseline_prefix + "/data"
baseline_results_prefix = baseline_prefix + "/results"

baseline_data_uri = "s3://{}/{}".format(bucket, baseline_data_prefix)
baseline_results_uri = "s3://{}/{}".format(bucket, baseline_results_prefix)
print("Baseline data uri: {}".format(baseline_data_uri))
print("Baseline results uri: {}".format(baseline_results_uri))

my_default_monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
)

my_default_monitor.suggest_baseline(
    baseline_dataset=baseline_data_uri + "/training-dataset-with-header.csv",
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=baseline_results_uri,
    wait=True,
    logs=False,
)

s3_client = boto3.Session().client("s3")
result = s3_client.list_objects(Bucket=bucket, Prefix=baseline_results_prefix)
report_files = [report_file.get("Key") for report_file in result.get("Contents")]
print("Found Files:")
print("\n ".join(report_files))

statistics_path = "{}/statistics.json".format(baseline_results_uri)
constraints_path = "{}/constraints.json".format(baseline_results_uri)

job_config = CheckJobConfig(role=role)
data_quality_config = DataQualityCheckConfig(
    baseline_dataset=transform_input_param,
    dataset_format=DatasetFormat.csv(header=False),
    output_s3_uri=s3_report_path,
)

from sagemaker.workflow.monitor_batch_transform_step import MonitorBatchTransformStep

transform_and_monitor_step = MonitorBatchTransformStep(
    name="MonitorCustomerChurnDataQuality",
    transform_step_args=transform_arg,
    monitor_configuration=data_quality_config,
    check_job_configuration=job_config,
    # since this is for data quality monitoring,
    # you could choose to run the monitoring job before the batch inference.
    monitor_before_transform=True,
    # if violation is detected in the monitoring, you can skip it and continue running batch transform
    fail_on_violation=False,
    supplied_baseline_statistics=statistics_path,
    supplied_baseline_constraints=constraints_path,
)

from sagemaker.workflow.pipeline import Pipeline

pipeline = Pipeline(
    name="MonitorDataQualityBatchTransformPipeline",
    parameters=[transform_input_param],
    steps=[create_model_step, transform_and_monitor_step],
)

pipeline.upsert(role_arn=role)

execution = pipeline.start()