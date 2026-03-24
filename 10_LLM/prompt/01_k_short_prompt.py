# Acknowledgement:github.com/sweetkruts/cs146s

import os
from dotenv import load_dotenv
# from ollama import chat
import openai

load_dotenv()
client = openai.OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")
    
NUM_RUNS_TIMES = 5

# TODO: Fill this in!
YOUR_SYSTEM_PROMPT_ = """
You are a helpful assistant that reverses the order of letters in a word. To reverse a word, take each letter and flip their positions completely - the first becomes last, second becomes second-to-last, etc.

Examples:

Word: "deepseek" 
Reversed: "keespeed"

Word: "nihao" 
Reversed: "oahin"

Word: "keepcoding"
Reversed: "gnidocpeek"


Process: Take the letters from right to left and write them in that order.

Only output the reversed word, nothing else.

"""

YOUR_SYSTEM_PROMPT = ''''
 你是一个程序高手，能够编写代码来解决问题。请根据用户的需求编写代码，并且只输出代码，不要任何其他文本。
'''
NONE_SYSTEM_PROMPT = '''
'''
USER_PROMPT = """
Reverse the order of letters in the following word. Only output the reversed word, no other text:

httpstatus
"""


EXPECTED_OUTPUT = "sutatsptth"

def test_your_prompt(system_prompt: str) -> bool:
    """Run the prompt up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            stream=False
        )
        output_text = response.choices[0].message.content.strip()
        print(f"Output: {output_text}")
        if output_text.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            # return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {output_text}")
            print("FAILURE")
            return False
    print("SUCCESS: All runs completed without failure.")
    return True

if __name__ == "__main__":
    """
    NONE: No system prompt. The model only sees the user prompt.
        all 5 wrong
    YOUR_SYSTEM_PROMPT: 第二次成功，第一次失败

    YOUR_SYSTEM_PROMPT_: 第一次成功
    例子比较重要，3个例子成功率比较高
    """
    test_your_prompt(YOUR_SYSTEM_PROMPT_)