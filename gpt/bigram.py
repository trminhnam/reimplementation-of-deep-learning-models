from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: Optional[int] = None):
        super().__init__()

        self.hidden_dim = hidden_dim

        if hidden_dim is None:
            self.input_embedding = nn.Embedding(vocab_size, vocab_size)
        else:
            self.input_embedding = nn.Embedding(vocab_size, hidden_dim)
            self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        logits = self.input_embedding(x)

        if self.hidden_dim is not None:
            logits = F.relu(self.lm_head(logits))

        if y is not None:  # Calculate the loss if labels are given
            # Flatten the logits and labels
            logits = logits.reshape(-1, logits.shape[-1]).contiguous()
            y = y.reshape(-1)
            loss = F.cross_entropy(logits, y)
            return loss, logits
        else:  # Return the logits for prediction
            return logits

    def generate(self, x: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(x)  # (batch_size, seq_len, vocab_size)

                # Calculate the probabilities of last token
                probs = F.softmax(logits[:, -1], dim=-1)

                # Sample the next token
                new_ids = torch.multinomial(probs, num_samples=1)

                # Concatenate the new token to the input
                x = torch.cat([x, new_ids], dim=-1)

        return x


if __name__ == "__main__":
    vocab_size = 100
    print(f"Vocab size: {vocab_size}")

    m = BigramLM(vocab_size)
    print(m)

    sequences = torch.randint(0, vocab_size, (3, 5))
    print(f"Training sequences:\n{sequences}")

    x = sequences[:, :-1]
    y = sequences[:, 1:]
    print(f"Input: {x.shape}, Target: {y.shape}")

    loss, logits = m(x, y)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item()}")

    # Generate new sequences
    x = torch.randint(0, vocab_size, (3, 5))
    n = 10
    generated = m.generate(x, n)
    print(f"Generated sequences:\n{generated}")
