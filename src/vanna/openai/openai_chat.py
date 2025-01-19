import os

import requests
from openai import OpenAI

from ..base import VannaBase


class OpenAI_Chat(VannaBase):
    def __init__(self, client=None, config=None):
        VannaBase.__init__(self, config=config)

        if not "x_customer" in config:
            raise Exception("XCustomer is required in the config")
        self.x_customer = config["x_customer"]

        # default parameters - can be overrided using config
        self.temperature = 0.7

        if "temperature" in config:
            self.temperature = config["temperature"]

        if "api_type" in config:
            raise Exception(
                "Passing api_type is now deprecated. Please pass an OpenAI client instead."
            )

        if "api_base" in config:
            raise Exception(
                "Passing api_base is now deprecated. Please pass an OpenAI client instead."
            )

        if "api_version" in config:
            raise Exception(
                "Passing api_version is now deprecated. Please pass an OpenAI client instead."
            )

        if client is not None:
            self.client = client
            return

        if config is None and client is None:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return

        if "api_key" in config:
            self.client = OpenAI(api_key=config["api_key"])

    def system_message(self, message: str) -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        return {"role": "assistant", "content": message}

    def update_token_usage(self, token_usage: int) -> any:
        url = self.x_customer.licenseUpdateUrl

        # Replace with actual values
        data = {
            "usage": str(int(token_usage)),
            "workspaceId": self.x_customer.workspaceId,
            # type 1 = openai chat token
            # type 2 = openai embedding token
            "type": 1
        }

        headers = {
            "Content-Type": "application/json",
            "X-Customer": self.x_customer.json()
        }

        response = requests.post(url, json=data, headers=headers)

        if not response.status_code == 200:
            print("Error in updating usage:", response.status_code, response.text)
            raise Exception(f"Error in updating usage: {response.status_code} {response.text}")

    def submit_prompt(self, prompt, **kwargs) -> str:
        if prompt is None:
            raise Exception("Prompt is None")

        if len(prompt) == 0:
            raise Exception("Prompt is empty")

        # Count the number of tokens in the message log
        # Use 4 as an approximation for the number of characters per token
        num_tokens = 0
        for message in prompt:
            num_tokens += len(message["content"]) / 4

        self.update_token_usage(num_tokens)

        if kwargs.get("model", None) is not None:
            model = kwargs.get("model", None)
            # print(
            #     f"Using model {model} for {num_tokens} tokens (approx)"
            # )
            response = self.client.chat.completions.create(
                model=model,
                messages=prompt,
                stop=None,
                temperature=self.temperature,
            )
        elif kwargs.get("engine", None) is not None:
            engine = kwargs.get("engine", None)
            # print(
            #     f"Using model {engine} for {num_tokens} tokens (approx)"
            # )
            response = self.client.chat.completions.create(
                engine=engine,
                messages=prompt,
                stop=None,
                temperature=self.temperature,
            )
        elif self.config is not None and "engine" in self.config:
            # print(
            #     f"Using engine {self.config['engine']} for {num_tokens} tokens (approx)"
            # )
            response = self.client.chat.completions.create(
                engine=self.config["engine"],
                messages=prompt,
                stop=None,
                temperature=self.temperature,
            )
        elif self.config is not None and "model" in self.config:
            # print(
            #     f"Using model {self.config['model']} for {num_tokens} tokens (approx)"
            # )
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=prompt,
                stop=None,
                temperature=self.temperature,
            )
        else:
            if num_tokens > 3500:
                model = "gpt-3.5-turbo-16k"
            else:
                model = "gpt-3.5-turbo"

            # print(f"Using model {model} for {num_tokens} tokens (approx)")
            response = self.client.chat.completions.create(
                model=model,
                messages=prompt,
                stop=None,
                temperature=self.temperature,
            )

        # Find the first response from the chatbot that has text in it (some responses may not have text)
        for choice in response.choices:
            if "text" in choice:
                return choice.text

        # If no response with text is found, return the first response's content (which may be empty)
        return response.choices[0].message.content
