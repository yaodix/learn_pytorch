from smolagents import CodeAgent,DuckDuckGoSearchTool, TransformersModel,load_tool,tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from openai import OpenAI
import os
from Gradio_UI import GradioUI
import gradio as gr
from smolagents import OpenAIServerModel

final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = OpenAIServerModel(model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            api_base="https://api.siliconflow.cn/v1",
            api_key=os.getenv("SILICONFLOW_API_KEY"))

# Import tool from Hub
# image_generation_tool = load_tool("/home/yao/myproject/learn_pytorch/model/openjourney", trust_remote_code=True)

with open("/home/yao/myproject/learn_pytorch/12_Agent/First_agent_template/prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model= model,
    tools=[final_answer, DuckDuckGoSearchTool()], ## add your tools here (don't remove final answer)
    max_steps=5,
    verbosity_level=1,
    # grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates,
     additional_authorized_imports=['datetime']
)


GradioUI(agent).launch()