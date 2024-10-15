import time
from typing import Union, Literal
from threading import Thread, Event

from llama_cpp import Llama
from llama_cpp.llama_types import (
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestAssistantMessage,
)
from capy.core.asr import CapyASR


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
        # Only CPU for now
        self.device = "cpu"
        self.compute_type = "int8"

        # LLM's internal state
        self.n_ctx = n_ctx
        self.batch_size = batch_size
        self.messages = [self.sys_prompt]  # Chat history buffer
        self.llm = None

        # Threads' events
        self.stop_generation_event = Event()  # Event to the generation task

        # ASR
        self.asr = CapyASR()

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

    def reset_llm(self):
        """
        Reset the chat engine to its initial state.

        This method will reset the chat history and the LLM's internal state.
        """
        self.messages = []
        self.llm.reset()

    def threaded_generate(self, cli: bool = False):
        """
        Thread task to run the chat generation. Cancel it anytime by sending stop_event.set()

        Args:
            cli (bool, optional): Run the LLM in CLI mode. Defaults to False.
        """
        assistant_response = ""
        if cli:
            # Print the prompt color
            print("\033[93mAssistant: \033[0m", end="", flush=True)

        for response in self.llm.create_chat_completion(
            messages=self.messages, stream=True
        ):
            if self.stop_generation_event.is_set():
                # Cancel the generation when the stop event is set
                break

            if "error" in response:
                # Print the error if any
                print(f"Error during streaming: {response['error']}")
                break

            if "choices" in response and response["choices"]:
                content = response["choices"][0]["delta"].get("content", "")
                if content:
                    # Concat the message chunks
                    assistant_response += content
                    if cli:
                        # Print the new content to the console
                        print(content, end="", flush=True)
        if cli:
            print()
        # Append the assistant response to the messages
        self.messages.append(
            ChatCompletionRequestAssistantMessage(
                role="assistant", content=assistant_response
            )
        )

    def get_text_input(self, cli=False):
        if cli:
            return input("\033[92mUser: \033[0m")
    
    def get_audio_input(self, cli=False):
        self.asr.listen_in_background()
        
        transcription = ""
        if cli:
            print("\033[92mUser: \033[0m", end="", flush=True)
        while True:
            try:
                transcript_chunk = self.asr.transcribe()
                if not transcript_chunk:
                    time.sleep(0.05)
                    continue
                if cli:
                    print(transcript_chunk, end="", flush=True)
                transcription += transcript_chunk
            except KeyboardInterrupt:
                self.asr.stop_transcribing()
                print()
                break
        return transcription


    def run_interactive_cli(self):
        try:
            if self.llm is None:
                raise ValueError("No model loaded!")

            print(
                "\n".join(
                    [
                        "Interactive CLI:",
                        "- While LLM responding, press Ctrl+C to stop the LLM's response generation process.",
                        "- Type 'quit', 'exit', or 'bye' to exit. Press Ctrl+C while not generating anything to exit as well.",
                        "- Voice input: Please speak into the microphone, stop recording by pressing Ctrl+C.",
                        "- Text input:  Enter to send your prompt"
                    ]
                )
            )
            
            while True:
                input_mode = input("\033[95mSystem: \033[0mEnter 't' for text input or 'v' for voice input: ").lower()
                user_input = self.get_text_input(cli=True) if input_mode == 't' else self.get_audio_input(cli=True)
                                
                if not user_input.strip():
                    continue
                if user_input.lower() in ["quit", "exit", "bye"]:
                    break

                self.messages.append(
                    ChatCompletionRequestUserMessage(role="user", content=user_input)
                )

                self.stop_generation_event.clear()

                generation_thread = Thread(
                    target=self.threaded_generate, kwargs={"cli": True}
                )
                generation_thread.start()

                try:
                    generation_thread.join()
                except KeyboardInterrupt:
                    self.stop_generation_event.set()
                    time.sleep(0.05)  # Give time to the generation to stop
        except KeyboardInterrupt:
            print("\nExiting...")
