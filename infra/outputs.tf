output "raw_bucket" {
  value = aws_s3_bucket.this["raw_data"].bucket
}

output "model_artifacts_bucket" {
  value = aws_s3_bucket.this["model_artifacts"].bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.inference.repository_url
}

output "sagemaker_endpoint_name" {
  value = aws_sagemaker_endpoint.champion.name
}
