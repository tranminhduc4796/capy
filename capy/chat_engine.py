import time
from typing import Union, Literal
from threading import Thread, Event

from llama_cpp import Llama
from llama_cpp.llama_types import (
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestAssistantMessage,
)


MODEL_PATH_MAP = {"llama-3": "./models/Llama-3.2-3B-Instruct-Q5_K_S.gguf"}


class CapyChatEngine:
    """
    Capy's chat engine manage the LLM and its behavior.
    Features:
    - Interactive conversations: Allow users to intervene in the LLM's response generation process, making more natural and dynamic conversations.
    - STT and TTS: Add audio and text-to-speech capabilities to your LLMs.
    """

    def __init__(
        self,
        sys_prompt: str = "You are a helpful assistant",
        n_ctx: int = 4096,
        batch_size: int = 512,
    ):
        """
        Initialize the chat engine.

        Args:
            sys_prompt (str): The system prompt to use for generating responses.
            n_ctx (int): The maximum context length (in tokens) that the model can process. Warning: Increase this can cause OOM. Defaults to 4096.
            batch_size (int): Number of tokens in a batch. Should be smaller then n_ctx and your RAM capacity. Defaults to 512.
        """
        self.sys_prompt = ChatCompletionRequestSystemMessage(
            role="system", content=sys_prompt
        )
        self.n_ctx = n_ctx
        self.batch_size = batch_size
        self.messages = [
            self.sys_prompt
        ]  # Chat history buffer, system prompt always included first
        self.llm = None
        self.stop_event = Event()  # Event for stopping the generation task

    def load_llm(self, model_name_or_path: Union[str, Literal["llama-3"]] = "llama-3"):
        """
        Load the LLM model.

        Args:
            model_name_or_path (Union[str, Literal["llama-3"]]): The name or path of the model to load. If a name is given, the path will be looked up in MODEL_PATH_MAP. Defaults to "llama-3".
        """
        if model_name_or_path in MODEL_PATH_MAP:
            model_name_or_path = MODEL_PATH_MAP[model_name_or_path]
        self.llm = Llama(model_path=model_name_or_path, n_ctx=self.n_ctx, verbose=False)

    def offload_llm(self):
        """
        Offload the LLM to release memory.

        This method will release the memory used by the LLM and set it to None.
        """
        self.llm.close()
        self.llm = None

    def reset(self):
        """
        Reset the chat engine to its initial state.

        This method will reset the chat history and the LLM's internal state.
        """
        self.messages = []
        self.llm.reset()

    def generate_task(self, cli_mode: bool = False):
        """
        Run the generation in a separate thread, allowing to cancel the generation anytime by sending stop_event.set()
        """
        assistant_response = ""
        if cli_mode:
            print("\033[93mAssistant: \033[0m", end="", flush=True)

        for response in self.llm.create_chat_completion(
            messages=self.messages, stream=True
        ):
            if self.stop_event.is_set():
                break
            if "error" in response:
                print(f"Error during streaming: {response['error']}")
                break
            if "choices" in response and response["choices"]:
                content = response["choices"][0]["delta"].get("content", "")
                if content:
                    assistant_response += content
                    if cli_mode:
                        print(content, end="", flush=True)
        if cli_mode:
            print()
        self.messages.append(
            ChatCompletionRequestAssistantMessage(
                role="assistant", content=assistant_response
            )
        )
    def run_interactive_cli(self):
        if self.llm is None:
            raise ValueError("No model loaded!")

        print("\n".join([
            "Interactive CLI:",
            "- Enter to send your prompt",
            "- While LLM responding, press Ctrl+C to stop the LLM's response generation process.",
            "- Type 'quit', 'exit', or 'bye' to exit. Press Ctrl+C while not generating anything to exit as well."
            ])
            )
        
        while True:
            user_input = input("\033[92mUser: \033[0m")
            if not user_input.strip():
                continue
            if user_input.lower() in ["quit", "exit", "bye"]:
                break

            self.messages.append(
                ChatCompletionRequestUserMessage(role="user", content=user_input)
            )

            self.stop_event.clear()

            generation_thread = Thread(target=self.generate_task, kwargs={"cli_mode": True})
            generation_thread.start()

            try:
                generation_thread.join()
            except KeyboardInterrupt:
                self.stop_event.set()
                time.sleep(0.05)  # Give time to the generation to stop