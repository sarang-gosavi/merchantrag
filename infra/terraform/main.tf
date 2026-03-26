# infra/terraform/main.tf
# MerchantRAG — AWS + Snowflake infrastructure as code
#
# Resources:
#   - AWS Lambda function (RAG API — serverless)
#   - API Gateway HTTP API (routes to Lambda)
#   - S3 bucket (data staging + archival)
#   - EventBridge rule (nightly pipeline trigger)
#   - SNS topic (validation failure alerts)
#   - CloudWatch log group + dashboard
#   - IAM roles
#
# Usage:
#   terraform init
#   terraform plan -var-file=terraform.tfvars
#   terraform apply

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    snowflake = {
      source  = "Snowflake-Labs/snowflake"
      version = "~> 0.87"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
  backend "s3" {
    bucket = "merchantrag-terraform-state"
    key    = "merchantrag/terraform.tfstate"
    region = "us-east-1"
  }
}

# ── Variables ─────────────────────────────────────────────────────────────────

variable "aws_region"         { default = "us-east-1" }
variable "environment"        { default = "prod" }
variable "anthropic_api_key"  { sensitive = true }
variable "pinecone_api_key"   { sensitive = true }
variable "snowflake_account"  {}
variable "snowflake_user"     {}
variable "snowflake_password" { sensitive = true }
variable "langchain_api_key"  { sensitive = true; default = "" }
variable "alert_email"        { default = "" }

# ── Providers ─────────────────────────────────────────────────────────────────

provider "aws" {
  region = var.aws_region
}

provider "snowflake" {
  account  = var.snowflake_account
  username = var.snowflake_user
  password = var.snowflake_password
  role     = "SYSADMIN"
}

# ── S3 bucket (staging + archive) ─────────────────────────────────────────────

resource "aws_s3_bucket" "merchantrag_data" {
  bucket = "merchantrag-data-${var.environment}"
  tags   = { Project = "merchantrag", Env = var.environment }
}

resource "aws_s3_bucket_versioning" "merchantrag_data" {
  bucket = aws_s3_bucket.merchantrag_data.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_lifecycle_configuration" "merchantrag_data" {
  bucket = aws_s3_bucket.merchantrag_data.id
  rule {
    id     = "archive-old-staging"
    status = "Enabled"
    filter { prefix = "archive/" }
    expiration { days = 90 }
  }
}

# ── Lambda function package ───────────────────────────────────────────────────

data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.root}/../"
  output_path = "${path.root}/merchantrag_lambda.zip"
  excludes = [
    ".git", "__pycache__", "*.pyc", ".env",
    "data/merchantrag.sqlite", "infra/terraform",
    "tests", "notebooks", ".venv",
  ]
}

# ── IAM role ──────────────────────────────────────────────────────────────────

resource "aws_iam_role" "lambda_exec" {
  name = "merchantrag-lambda-exec"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_s3" {
  name = "merchantrag-lambda-s3"
  role = aws_iam_role.lambda_exec.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket"]
      Resource = [
        aws_s3_bucket.merchantrag_data.arn,
        "${aws_s3_bucket.merchantrag_data.arn}/*"
      ]
    }]
  })
}

# ── Lambda function ───────────────────────────────────────────────────────────

resource "aws_lambda_function" "merchantrag_query" {
  filename         = data.archive_file.lambda_zip.output_path
  function_name    = "merchantrag-query-${var.environment}"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "infra.lambda_handler.handler"
  runtime          = "python3.11"
  timeout          = 60
  memory_size      = 1024
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256

  environment {
    variables = {
      ANTHROPIC_API_KEY     = var.anthropic_api_key
      PINECONE_API_KEY      = var.pinecone_api_key
      SNOWFLAKE_ACCOUNT     = var.snowflake_account
      SNOWFLAKE_USER        = var.snowflake_user
      SNOWFLAKE_PASSWORD    = var.snowflake_password
      LANGCHAIN_API_KEY     = var.langchain_api_key
      LANGCHAIN_TRACING_V2  = var.langchain_api_key != "" ? "true" : "false"
      LANGCHAIN_PROJECT     = "merchantrag-${var.environment}"
      USE_LOCAL_VECTORS     = "false"
      VECTOR_BACKEND        = "pinecone"
      USE_LOCAL_DB          = "false"
      KAFKA_MOCK_MODE       = "true"
      USE_SAMPLE_DATA       = "false"
      AWS_S3_BUCKET         = aws_s3_bucket.merchantrag_data.id
      USE_LOCAL_STORAGE     = "false"
      ENVIRONMENT           = var.environment
    }
  }

  tags = { Project = "merchantrag", Env = var.environment }
}

resource "aws_lambda_alias" "live" {
  name             = "live"
  function_name    = aws_lambda_function.merchantrag_query.function_name
  function_version = "$LATEST"
}

# ── API Gateway (HTTP API) ────────────────────────────────────────────────────

resource "aws_apigatewayv2_api" "merchantrag" {
  name          = "merchantrag-api-${var.environment}"
  protocol_type = "HTTP"
  cors_configuration {
    allow_headers = ["Content-Type", "Authorization"]
    allow_methods = ["GET", "POST", "DELETE", "OPTIONS"]
    allow_origins = ["*"]
    max_age       = 300
  }
}

resource "aws_apigatewayv2_stage" "prod" {
  api_id      = aws_apigatewayv2_api.merchantrag.id
  name        = var.environment
  auto_deploy = true
  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_logs.arn
  }
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.merchantrag.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.merchantrag_query.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "query" {
  api_id    = aws_apigatewayv2_api.merchantrag.id
  route_key = "POST /query"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_route" "ingest" {
  api_id    = aws_apigatewayv2_api.merchantrag.id
  route_key = "POST /ingest"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.merchantrag.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.merchantrag_query.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.merchantrag.execution_arn}/*/*"
}

# ── EventBridge — nightly pipeline trigger ────────────────────────────────────

resource "aws_cloudwatch_event_rule" "nightly_ingest" {
  name                = "merchantrag-nightly-ingest"
  description         = "Trigger MerchantRAG pipeline reload every night at 2am UTC"
  schedule_expression = "cron(0 2 * * ? *)"
}

resource "aws_cloudwatch_event_target" "nightly_ingest" {
  rule      = aws_cloudwatch_event_rule.nightly_ingest.name
  target_id = "merchantrag-nightly"
  arn       = aws_lambda_function.merchantrag_query.arn
  input     = jsonencode({ path = "/pipeline/reload", httpMethod = "POST", body = "{}" })
}

resource "aws_lambda_permission" "eventbridge" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.merchantrag_query.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.nightly_ingest.arn
}

# ── S3 trigger — ingest on file arrival ───────────────────────────────────────

resource "aws_s3_bucket_notification" "staging_trigger" {
  bucket = aws_s3_bucket.merchantrag_data.id
  lambda_function {
    lambda_function_arn = aws_lambda_function.merchantrag_query.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "staging/"
  }
  depends_on = [aws_lambda_permission.api_gateway]
}

# ── CloudWatch logs + dashboard ───────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/merchantrag-query-${var.environment}"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "api_logs" {
  name              = "/aws/apigateway/merchantrag-${var.environment}"
  retention_in_days = 14
}

resource "aws_cloudwatch_metric_alarm" "validation_failures" {
  alarm_name          = "merchantrag-validation-failures"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "MerchantRAG Lambda errors > 5 in 5 min — data validation may be failing"
  dimensions          = { FunctionName = aws_lambda_function.merchantrag_query.function_name }
  alarm_actions       = var.alert_email != "" ? [aws_sns_topic.alerts[0].arn] : []
}

resource "aws_sns_topic" "alerts" {
  count = var.alert_email != "" ? 1 : 0
  name  = "merchantrag-alerts-${var.environment}"
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alerts[0].arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ── Snowflake resources ───────────────────────────────────────────────────────

resource "snowflake_database" "merchantrag" {
  name    = "MERCHANTRAG_DB"
  comment = "MerchantRAG — merchant intelligence data platform"
}

resource "snowflake_schema" "raw" {
  database = snowflake_database.merchantrag.name
  name     = "RAW"
}

resource "snowflake_schema" "staging" {
  database = snowflake_database.merchantrag.name
  name     = "STAGING"
}

resource "snowflake_schema" "mart" {
  database = snowflake_database.merchantrag.name
  name     = "MART"
}

resource "snowflake_warehouse" "compute" {
  name           = "MERCHANTRAG_WH"
  warehouse_size = "X-SMALL"
  auto_suspend   = 60
  auto_resume    = true
  comment        = "MerchantRAG compute warehouse"
}

resource "snowflake_stage" "merchant_stage" {
  name        = "MERCHANT_STAGE"
  database    = snowflake_database.merchantrag.name
  schema      = snowflake_schema.raw.name
  url         = "s3://${aws_s3_bucket.merchantrag_data.id}/snowpipe/"
  comment     = "Snowpipe staging — merchant transaction files"
}

# ── Outputs ───────────────────────────────────────────────────────────────────

output "api_endpoint" {
  value       = "${aws_apigatewayv2_stage.prod.invoke_url}"
  description = "MerchantRAG API Gateway endpoint"
}

output "lambda_function_name" {
  value = aws_lambda_function.merchantrag_query.function_name
}

output "s3_bucket" {
  value = aws_s3_bucket.merchantrag_data.id
}

output "snowflake_database" {
  value = snowflake_database.merchantrag.name
}
