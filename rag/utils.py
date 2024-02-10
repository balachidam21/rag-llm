import os
import subprocess
import openai
import tiktoken

def get_num_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def trim(text, max_context_length):
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(enc.encode(text)[:max_context_length])

def execute_bash(command):
    results = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return results

def get_client(llm):
    if llm.startswith("gpt"):
        base_url = os.environ["OPENAI_API_BASE"]
        api_key = os.environ["OPEN_AI_KEY"]
    else:
        base_url = os.environ["ANYSCALE_API_BASE"]
        api_key = os.environ["ANYSCALE_API_KEY"]
    client = openai.OpenAI(api_key=api_key)
    print(client)
    return client
