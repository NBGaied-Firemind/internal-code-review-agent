
MODEL_IDS = {
    "sonnet-3.5": {
        "id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "arn" :"arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0",
        "input_cost": 0.003,
        "output_cost": 0.015
    },
    "haiku": {
        "id": "anthropic.claude-3-haiku-20240307-v1:0",
        "input_cost": 0.00025,
        "output_cost": 0.00125
    }, 
    "nova-micro": {
        "id": "amazon.nova-micro-v1:0",
        "arn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-micro-v1:0",
        "input_cost": 0.000035,
        "output_cost": 0.00014
    }, 
    "nova-pro": {
        "id": "amazon.nova-pro-v1:0",
        "arn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0",
        "input_cost": 0.0008,
        "output_cost": 0.0032
    }
}