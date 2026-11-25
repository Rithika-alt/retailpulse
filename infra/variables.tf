variable "aws_region" {
  description = "AWS region (never applied; documentation only)."
  type        = string
  default     = "us-east-1"
}

variable "raw_bucket_name" {
  description = "Raw sales data bucket. Matches the bucket LocalStack simulates locally. Must be globally unique on real AWS."
  type        = string
  default     = "retailpulse-raw-data"
}

variable "model_bucket_name" {
  description = "Bucket for packaged model artifacts (model.tar.gz). Must be globally unique on real AWS."
  type        = string
  default     = "retailpulse-model-artifacts"
}

variable "image_tag" {
  description = "Tag of the inference image in ECR (repository tags are immutable, so no 'latest')."
  type        = string
  default     = "v1"
}

variable "model_artifact_key" {
  description = "PLACEHOLDER key of the packaged champion model in the artifacts bucket; nothing uploads it."
  type        = string
  default     = "models/retailpulse-champion/model.tar.gz"
}

variable "instance_type" {
  description = "SageMaker instance type for the real-time endpoint."
  type        = string
  default     = "ml.t2.medium"
}
