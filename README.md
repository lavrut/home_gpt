# home_gpt

This is a personal project.

Considerations: Assumes Hugging Face GPT2 model has been enhanced to scale the size of data it can process, the number of parameters it can handle, and includes additional enhancements for sparse attention, reversible layers, and activation checkpointing. Currently, GPT2 model is not capable of supporting the scaling required for these enhancements.

No data is used in this model.

This app assumes the use of AWS SSO. 

To get this model and application built you will need to setup the environment. 

Tech stack:
- Amazon SageMaker
- Amazon S3
- AWS Cognito - handles user authentication and authorization
- AWS IAM - handles user access to resources
- Amazon EC2 (for instances that use NVIDIA A100 GPUs (e.g. p4d.24xlarge))
- Amazon CloudWatch: A monitoring and observability service 
- AWS Cost Explorer: You can use this to monitor and optimize your spending on the resources used for the LLM model.
- Hugging Face's GPT2 transformer library
