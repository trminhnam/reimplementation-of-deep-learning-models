import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer


class Llama:

    def __init__(
        self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(
        checkpoints_dir: str,
        tokenizer_path: str,
        load_model: bool,
        max_seq_len: int,
        max_batch_size: int,
        device: str,
    ):
        start_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, "No checkpoint found"

            chk_path = checkpoints[0]
            print(f"Loading checkpoint {chk_path}")
            checkpoint = torch.load(chk_path, map_location="cpu")
            print(f"Loaded checkpoit in {(time.time() - start_time):.2f}s")
            start_time = time.time()

        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_length=max_seq_len, max_batch_size=max_batch_size, device=device, **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {(time.time() - start_time):.2f}s")

        return Llama(model, tokenizer, model_args)

    def text_completion(
        self,
        prompts: list[str],
        temperature: float = 0.6,
        top_p: float = 0.6,
        max_gen_len: Optional[int] = None,
    ):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_length - 1

        prompt_tokens = [
            self.tokenizer.encode(p, out_type=int, add_bos=True, add_eos=False) for p in prompts
        ]
        batch_size = len(prompts)
        assert batch_size <= self.args.max_batch_size

        max_prompt_len = max(len(p) for p in prompt_tokens)
        total_len = min(self.args.max_seq_length, max_gen_len + max_prompt_len)

        # Create a matrices that contains the generated tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):  # Fill the matrix with each prompt ids
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        # Used to know if the end of the video has been reached to stop generating tokens for that sequence
        eos_reached = torch.tensor([False] * batch_size, device=device)

        # A mask matrices that True if the token is not a padding token and False otherwise
        # If True (not a padding token) then the token is forward through the model to build the cache
        # If False (padding token) then the token is forward through the model and get the logits to predict the next token
        prompt_tokens_mask = tokens != pad_id
        for cur_pos in tqdm(range(1, total_len), desc="Generating tokens"):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos - 1 : cur_pos], cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.view(-1)

            # Replace the padding token with the next token, otherwise keep the token as it is
            # If it is a prompt token passed through the model to build the cache, then return the prompt token
            # Otherwise return the next token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)

            # Update the matrix with the new token
            tokens[:, cur_pos] = next_token

            # Check and update EOS reached
            # The generated token is EOS and mask is False (not a prompt token) then set EOS reached to True
            eos_reached = eos_reached | (
                ~prompt_tokens_mask[:, cur_pos] & (next_token == self.tokenizer.eos_id())
            )

            # If all EOS reached are True then break the loop
            if all(eos_reached):
                break

        # Decode the generated tokens one prompt in the batch at a time
        out_tokens = []
        out_text = []
        for prompt_ids, cur_prompt_tokens in enumerate(tokens.tolist()):
            if self.tokenizer.eos_id() in cur_prompt_tokens:
                eos_idx = cur_prompt_tokens.index(
                    self.tokenizer.eos_id()
                )  # Find the first EOS token
                cur_prompt_tokens = cur_prompt_tokens[
                    :eos_idx
                ]  # Keep only the tokens before the EOS token
            out_tokens.append(cur_prompt_tokens)
            out_text.append(self.tokenizer.decode(cur_prompt_tokens))

        return (out_tokens, out_text)

    def _sample_top_p(self, probs, p):
        # Sort the probabilities in descending order, return the probabilities and the indices
        probs_sort, probs_idx = torch.sort(probs, descending=True)

        # Then taking the cumulative sum of the sorted probabilities
        probs_cum = torch.cumsum(probs_sort, dim=-1)

        # Then shift the sorted probabilities to the right by 1 and insert 0 at the beginning
        probs_shifted = torch.roll(probs_cum, 1, dims=-1)
        probs_shifted[..., 0] = 0.0

        # Then mask the probabilities that are above the threshold p (we want sum of probabilities to be just right greater than or equal to p)
        mask = probs_shifted > p
        probs_sort[mask] = 0.0

        # Divide the masked probs_sort by the sum of each row to get the new probabilities
        probs_sort /= probs_sort.sum(dim=-1, keepdim=True)

        # Then sample from the normalized probabilities
        new_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, new_token)

        return next_token


if __name__ == "__main__":

    torch.manual_seed(0)

    allow_cuda = False
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    model = Llama.build(
        checkpoints_dir="./Llama-2-7b",
        tokenizer_path="tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device,
    )

    print(f"Model loaded on {device}")

    prompts = [
        "Translate the following text to French: 'Hello, how are you?'",
        "Tell me the definition of machine learning in 1 sentence.",
        "What is the capital of France?",
    ]

    out_tokens, out_text = model.text_completion(prompts, max_gen_len=32)
    for p, t in zip(prompts, out_text):
        print(f"Prompt: {p}")
        print(f"Output: {t}")
        print()
