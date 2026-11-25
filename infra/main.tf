# =============================================================================
# STATUS: VALIDATED FOR SYNTAX/STRUCTURE ONLY - NEVER PLANNED OR APPLIED.
#
# This Terraform describes how RetailPulse's serving/storage layer WOULD be deployed
# on real AWS. It has only been checked with `terraform init -backend=false` and
# `terraform validate`. `terraform plan` / `terraform apply` were deliberately never
# run and no AWS credentials were ever configured, so that this portfolio project
# incurs zero AWS cost. Nothing described here exists in any AWS account.
#
# What actually runs today (docker compose): LocalStack S3 (stands in for the raw-data
# bucket below), a local FastAPI inference service (stands in for the endpoint), and a
# local MLflow server (the model registry).
#
# SageMaker HTTP contract: the inference image now implements it. It serves GET /ping and
# POST /invocations (same prediction code as /predict), listens on $PORT (set to 8080 below,
# as SageMaker requires), and ignores the "serve" argument SageMaker starts containers with.
# This was only exercised locally (docker-compose.sagemaker.yml), never on SageMaker itself.
#
# Known gaps before a real deploy (honest list):
#   * Model loading: SageMaker would extract model_data_url into /opt/ml/model, but the API
#     loads the model from the MLflow registry (MLFLOW_TRACKING_URI), which does not exist
#     inside SageMaker. A packaged-model loading mode is not implemented.
#   * model_data_url below is a PLACEHOLDER: nothing packages the MLflow "production"
#     model as model.tar.gz into the artifacts bucket.
#   * The image has never been pushed to ECR and the endpoint was never created.
#   * S3 bucket names are globally unique; the defaults will likely need changing.
# =============================================================================

provider "aws" {
  region = var.aws_region
  # No profile, keys or assume_role on purpose: this configuration is never applied.
}

locals {
  name = "retailpulse"
  tags = {
    Project   = "retailpulse"
    ManagedBy = "terraform"
  }
  buckets = {
    raw_data        = var.raw_bucket_name   # LocalStack simulates this bucket locally
    model_artifacts = var.model_bucket_name # would hold the packaged champion model
  }
}

# ---- S3 ----------------------------------------------------------------------
resource "aws_s3_bucket" "this" {
  for_each = local.buckets

  bucket = each.value
  tags   = merge(local.tags, { Purpose = each.key })
}

resource "aws_s3_bucket_versioning" "this" {
  for_each = aws_s3_bucket.this

  bucket = each.value.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = aws_s3_bucket.this

  bucket = each.value.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each = aws_s3_bucket.this

  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---- ECR: image for the inference API ------------------------------------------
resource "aws_ecr_repository" "inference" {
  name                 = "${local.name}-inference-api"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = local.tags
}

resource "aws_ecr_lifecycle_policy" "inference" {
  repository = aws_ecr_repository.inference.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ---- IAM: execution role for SageMaker ------------------------------------------
data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "sagemaker_access" {
  statement {
    sid     = "ReadModelArtifacts"
    actions = ["s3:GetObject", "s3:ListBucket"]
    resources = [
      aws_s3_bucket.this["model_artifacts"].arn,
      "${aws_s3_bucket.this["model_artifacts"].arn}/*",
    ]
  }
  statement {
    sid       = "PullInferenceImage"
    actions   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = [aws_ecr_repository.inference.arn]
  }
  statement {
    sid       = "EcrAuthToken"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:*:*:log-group:/aws/sagemaker/*"]
  }
}

resource "aws_iam_role" "sagemaker_execution" {
  name               = "${local.name}-sagemaker-execution"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy" "sagemaker_access" {
  name   = "${local.name}-sagemaker-access"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.sagemaker_access.json
}

# ---- SageMaker real-time endpoint -------------------------------------------------
# Conceptually: the model registered as `retailpulse-champion` (alias `production`) in MLflow.
resource "aws_sagemaker_model" "champion" {
  name               = "${local.name}-champion"
  execution_role_arn = aws_iam_role.sagemaker_execution.arn

  primary_container {
    image          = "${aws_ecr_repository.inference.repository_url}:${var.image_tag}"
    model_data_url = "s3://${aws_s3_bucket.this["model_artifacts"].bucket}/${var.model_artifact_key}" # PLACEHOLDER
    environment = {
      MLFLOW_MODEL_NAME  = "retailpulse-champion"
      MLFLOW_MODEL_ALIAS = "production"
      PORT               = "8080" # SageMaker requires the container to listen on 8080
    }
  }

  tags = local.tags
}

resource "aws_sagemaker_endpoint_configuration" "champion" {
  name = "${local.name}-champion-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.champion.name
    instance_type          = var.instance_type
    initial_instance_count = 1
    initial_variant_weight = 1
  }

  tags = local.tags
}

resource "aws_sagemaker_endpoint" "champion" {
  name                 = "${local.name}-champion"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.champion.name
  tags                 = local.tags
}
