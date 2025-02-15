import os
import json
import boto3
import logging
import requests

from mangum import Mangum
from string import Template
from bs4 import BeautifulSoup
from atlassian import Confluence
from requests.auth import HTTPBasicAuth
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from model_information import MODEL_IDS


PROFILE_NAME = os.getenv("PROFILE_NAME")
# "firemindsandbox"

REGION_NAME = os.getenv("REGION_NAME")
# "us-east-1"

CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
# page_id = '3539730433'

EMAIL = os.getenv("EMAIL")

CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")

CODE_BUCKET = os.getenv("CODE_BUCKET")
# "nour-code-review-agent-test-bucket"

origins = [os.getenv("ORIGIN")]

session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)
bedrock = session.client("bedrock-runtime")
s3 = session.client("s3")

logger_level = os.getenv("LOGGER_LEVEL", "INFO").upper()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def retrieve_code_file_from_s3(file_key, bucket_name=CODE_BUCKET):
    response = s3.get_object(Bucket=bucket_name, Key=file_key)
    file_content = response["Body"].read().decode("utf-8")
    return file_content


def get_confluence_page_content(page_id, confluence_url, email, api_token):
    url = f"{confluence_url}/rest/api/content/{page_id}?expand=body.storage"

    response = requests.get(
        url,
        auth=HTTPBasicAuth(email, api_token),
        headers={"Accept": "application/json"},
    )
    if response.status_code == 200:
        data = response.json()
        page_title = data.get("title")
        page_content = (
            data.get("body", {}).get("storage", {}).get("value", "No content found.")
        )
        return page_title, page_content
    else:
        return f"Error: {response.status_code} - {response.text}"


def extract_headers_and_content(content):
    content_dict = {}

    soup = BeautifulSoup(content, "html.parser")

    current_header = None
    current_content = []

    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "p"]):
        if tag.name in ["h1", "h2", "h3", "h4", "h5"]:
            if current_header:
                content_dict[current_header] = " ".join(current_content).strip()

            current_header = tag.get_text().strip()
            current_content = []

        elif tag.name == "p" and current_header:
            current_content.append(tag.get_text().strip())

    if current_header:
        content_dict[current_header] = " ".join(current_content).strip()

    return content_dict


def invoke_bedrock(model_id, messages, system_prompt, client=bedrock):

    temperature = 0.2
    top_p = 0

    inference_config = {"temperature": temperature, "maxTokens": 4000, "topP": top_p}

    response = client.converse(
        modelId=model_id,
        messages=messages,
        system=system_prompt,
        inferenceConfig=inference_config,
    )

    input_tokens = float(response["usage"]["inputTokens"])
    output_tokens = float(response["usage"]["outputTokens"])
    total_tokens = float(response["usage"]["totalTokens"])
    latency = float(response["metrics"]["latencyMs"])

    bedrock_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency": latency,
    }
    return response, bedrock_metadata


def format_message(role, content):
    return [{"role": role, "content": [{"text": content}]}]


def format_prompt(prompt):
    return [
        {
            "text": prompt,
        }
    ]


def generate_output(model_id, system_prompt, messages, client=bedrock):

    response, _ = invoke_bedrock(
        messages=messages, model_id=model_id, system_prompt=system_prompt
    )
    final_response = response["output"]["message"]["content"][0]["text"]

    return final_response


output_format = """
APIs: 
1. Use ESM JavaScript, aiming for the latest stable node version: Pass 
2. If you can, use LLRT (Optional): Fail
...
Data Pipelines: 
1. Prefer Python, aiming for the latest stable python version: Pass
...
"""

agent_backend_system_prompt_template = Template(
    """
You are a code review agent at Firemind, an AWS Advanced Partner specializing in AI and ML solutions. 
Your primary task is to review the backend code of Firemind developers to ensure it aligns with Firemind's technology choices guidelines.

Instructions:
- Review the code against Firemind's technology choices guidelines, using the guidelines as a checklist.
- For each guideline, determine whether it has been fully implemented or not. Provide a score of "Complete" or "Incomplete" for each.
- Return the results in the specified output format: $output_format.
- Use the header and bullet points from the guidelines to reference each specific point when providing your assessment.

Firemind's technology choice guidelines: 
$tech_choice_doc

Ensure your feedback is clear and aligned with the expectations for code quality and compliance with Firemind's standards.
"""
)

agent_frontend_system_prompt_template = Template(
    """
You are a code review agent at Firemind, an AWS Advanced Partner specializing in AI and ML solutions.  
Your primary task is to review the **frontend code** of Firemind developers to ensure it aligns with Firemind's technology choices guidelines.  

Instructions:
- Review the code against **Firemind's frontend technology choices guidelines**, using them as a checklist.  
- Assess **code quality, performance, accessibility, UI/UX best practices, and maintainability**.  
- For each guideline, determine whether it has been fully implemented or not. Provide a score of **"Complete" or "Incomplete"** for each.  
- Return the results in the specified output format: **$output_format**.  
- Use the **header and bullet points** from the guidelines to reference each specific point when providing your assessment.  

Firemind's technology choice guidelines: 
$tech_choice_doc

Ensure your feedback is clear and aligned with the expectations for code quality and compliance with Firemind's standards.
"""
)


def handle_code_review_request(page_id, file_key, system_prompt_template):
    # RETRIEVE CODE FILE FROM S3
    code_file = retrieve_code_file_from_s3(file_key)
    # RETRIEVE TECH STANDARDS FILE FROM CONFLUENCE AND FORMAT
    title, content = get_confluence_page_content(
        page_id, CONFLUENCE_URL, EMAIL, CONFLUENCE_API_TOKEN
    )
    tech_choice_doc = extract_headers_and_content(content)
    # FORMAT SYSTEM PROMPT
    system_prompt = system_prompt_template.substitute(
        output_format=output_format, tech_choice_doc=tech_choice_doc
    )
    formatted_message = format_message("user", code_file)
    formatted_prompt = format_prompt(system_prompt)
    # INVOKE BEDROCK
    return generate_output(
        MODEL_IDS["nova-micro"]["id"], formatted_prompt, formatted_message
    )


@app.post("/api/v0/BackendAgent")
async def backend_agent(req: Request):
    try:
        body = await req.json()
        logger.info(f"Body: {body}")
        page_id = body.get("page_id")
        file_key = body.get("file_key")

        # INVOKE BEDROCK
        response = handle_code_review_request(
            page_id, file_key, agent_backend_system_prompt_template
        )

        return response

    except Exception as e:
        logger.error(f"Error when invoking BackendAgent api: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.post("/api/v0/FrontendAgent")
async def frontend_agent(req: Request):
    try:
        body = await req.json()
        logger.info(f"Body: {body}")
        page_id = body.get("page_id")
        file_key = body.get("file_key")

        # INVOKE BEDROCK
        response = handle_code_review_request(
            page_id, file_key, agent_frontend_system_prompt_template
        )

        return response

    except Exception as e:
        logger.error(f"Error when invoking BackendAgent api: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong")


handler = Mangum(app)
