## Build Open Source Strands Agent using AWS Bedrock
Below steps were tested on MacOS.

```
# Check python version
python -V
Python 3.10.15

bash 
export GITHUB_TOKEN=ghp_000111222333xxx

# create virtual env
python -m venv .venv
.venv/bin/activate

# Install required packages
pip install strands-agents strands-agents-tools boto3

# Configure AWS Credentials
bash
# Configure a specific profile for Bedrock
aws configure --profile bedrock

# Verify your configuration
aws sts get-caller-identity --profile bedrock

# Check which models you have access to
aws bedrock list-foundation-models --profile bedrock


# model used: 
"anthropic.claude-3-5-sonnet-20240620-v1:0"

#
(.venv) bash-5.2$ python my_agent/agent.py 
DEBUG | strands.models.bedrock | config=<{'model_id': 'anthropic.claude-3-5-sonnet-20240620-v1:0'}> | initializing
DEBUG | botocore.hooks | Changing event name from creating-client-class.iot-data to creating-client-class.iot-data-plane
DEBUG | botocore.hooks | Changing event name from before-call.apigateway to before-call.api-gateway
DEBUG | botocore.hooks | Changing event name from request-created.machinelearning.Predict to request-created.machine-learning.Predict
DEBUG | botocore.hooks | Changing event name from before-parameter-build.autoscaling.CreateLaunchConfiguration to before-parameter-build.auto-scaling.CreateLaunchConfiguration
DEBUG | botocore.hooks | Changing event name from before-parameter-build.route53 to before-parameter-build.route-53
DEBUG | botocore.hooks | Changing event name from request-created.cloudsearchdomain.Search to request-created.cloudsearch-domain.Search
DEBUG | botocore.hooks | Changing event name from docs.*.autoscaling.CreateLaunchConfiguration.complete-section to docs.*.auto-scaling.CreateLaunchConfiguration.complete-section
DEBUG | botocore.hooks | Changing event name from before-parameter-build.logs.CreateExportTask to before-parameter-build.cloudwatch-logs.CreateExportTask
DEBUG | botocore.hooks | Changing event name from docs.*.logs.CreateExportTask.complete-section to docs.*.cloudwatch-logs.CreateExportTask.complete-section
DEBUG | botocore.hooks | Changing event name from before-parameter-build.cloudsearchdomain.Search to before-parameter-build.cloudsearch-domain.Search
DEBUG | botocore.hooks | Changing event name from docs.*.cloudsearchdomain.Search.complete-section to docs.*.cloudsearch-domain.Search.complete-section
DEBUG | botocore.utils | IMDS ENDPOINT: http://169.254.169.254/
DEBUG | botocore.credentials | Looking for credentials via: env
DEBUG | botocore.credentials | Looking for credentials via: assume-role
DEBUG | botocore.credentials | Looking for credentials via: assume-role-with-web-identity
DEBUG | botocore.credentials | Looking for credentials via: sso
DEBUG | botocore.credentials | Looking for credentials via: shared-credentials-file
INFO | botocore.credentials | Found credentials in shared credentials file: ~/.aws/credentials
DEBUG | botocore.loaders | Loading JSON file: /Users/user/Documents/repo/cloud/aws/amazon_strands/setup-agent/.venv/lib/python3.10/site-packages/botocore/data/endpoints.json
DEBUG | botocore.loaders | Loading JSON file: /Users/user/Documents/repo/cloud/aws/amazon_strands/setup-agent/.venv/lib/python3.10/site-packages/botocore/data/sdk-default-configuration.json
DEBUG | botocore.hooks | Event choose-service-name: calling handler <function handle_service_name_alias at 0x1109fcd30>
DEBUG | botocore.loaders | Loading JSON file: /Users/user/Documents/repo/cloud/aws/amazon_strands/setup-agent/.venv/lib/python3.10/site-packages/botocore/data/bedrock-runtime/2023-09-30/service-2.json.gz
DEBUG | botocore.loaders | Loading JSON file: /Users/user/Documents/repo/cloud/aws/amazon_strands/setup-agent/.venv/lib/python3.10/site-packages/botocore/data/bedrock-runtime/2023-09-30/endpoint-rule-set-1.json.gz
DEBUG | botocore.loaders | Loading JSON file: /Users/user/Documents/repo/cloud/aws/amazon_strands/setup-agent/.venv/lib/python3.10/site-packages/botocore/data/partitions.json
DEBUG | botocore.hooks | Event creating-client-class.bedrock-runtime: calling handler <function remove_bedrock_runtime_invoke_model_with_bidirectional_stream at 0x110a24310>
DEBUG | botocore.hooks | Event creating-client-class.bedrock-runtime: calling handler <function add_generate_presigned_url at 0x11097c9d0>
DEBUG | botocore.configprovider | Looking for endpoint for bedrock-runtime via: environment_service
DEBUG | botocore.configprovider | Looking for endpoint for bedrock-runtime via: environment_global
DEBUG | botocore.configprovider | Looking for endpoint for bedrock-runtime via: config_service
DEBUG | botocore.configprovider | Looking for endpoint for bedrock-runtime via: config_global
DEBUG | botocore.configprovider | No configured endpoint found.
DEBUG | botocore.regions | Creating a regex based endpoint for bedrock-runtime, us-east-1
DEBUG | botocore.endpoint | Setting bedrock-runtime timeout as (60, 60)
DEBUG | botocore.loaders | Loading JSON file: /Users/user/Documents/repo/cloud/aws/amazon_strands/setup-agent/.venv/lib/python3.10/site-packages/botocore/data/_retry.json
DEBUG | botocore.client | Registering retry handlers for service: bedrock-runtime
DEBUG | strands.models.bedrock | region=<us-east-1> | bedrock client created
DEBUG | strands.tools.registry | tool_name=<calculator>, tool_type=<function>, is_dynamic=<False> | registering tool
DEBUG | strands.tools.registry | tool_name=<current_time>, tool_type=<function>, is_dynamic=<False> | registering tool
DEBUG | strands.tools.loader | tool_path=</Users/user/Documents/repo/cloud/aws/amazon_strands/setup-agent/.venv/lib/python3.10/site-packages/strands_tools/python_repl.py> | loading python tool from path
DEBUG | strands.tools.registry | tool_name=<python_repl>, tool_type=<python>, is_dynamic=<True> | registering tool
DEBUG | strands.tools.registry | tool_name=<python_repl>, tool_type=<python> | skipping hot reloading
DEBUG | strands.tools.registry | tool_name=<letter_counter>, tool_type=<function>, is_dynamic=<False> | registering tool
DEBUG | strands.tools.registry | tools_dir=</Users/user/Documents/repo/cloud/aws/amazon_strands/setup-agent/tools> | found tools directory
DEBUG | strands.tools.registry | tools_dir=</Users/user/Documents/repo/cloud/aws/amazon_strands/setup-agent/tools> | scanning
DEBUG | strands.tools.registry | tool_modules=<[]> | discovered
DEBUG | strands.tools.registry | tool_count=<0>, success_count=<0> | finished loading tools
Running the agent...

Agent Response: