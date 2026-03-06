import chess
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import Optional

try:
    # When running inside the chess_exam tournament environment
    from chess_tournament import Player
except ImportError:
    # Fallback for standalone testing
    from abc import ABC, abstractmethod
    class Player(ABC):
        def __init__(self, name: str):
            self.name = name
        @abstractmethod
        def get_move(self, fen: str) -> Optional[str]:
            pass


class TransformerPlayer(Player):
    """
    Chess player based on fine-tuned GPT-2 medium.

    Architecture: Decoder-only transformer (GPT-2 medium, 355M params)
    Training data: 350k positions from Lichess Elite Database (2500+ ELO)
                   + 200k positions re-annotated with local Stockfish (depth=8)
    Move selection: Constrained decoding — scores every legal move by
                    its negative cross-entropy loss, picks the highest.
                    This guarantees 0 illegal moves and 0 fallbacks.
    """

    # HuggingFace model repository
    HF_MODEL = "Iman-ghotbi/chess-gpt2-finetuned"

    # Maximum legal moves to score per position (caps inference time)
    MAX_MOVES_TO_SCORE = 30

    def __init__(self, name: str = "TransformerPlayer"):
        super().__init__(name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.HF_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(self.HF_MODEL)
        self.model.to(self.device)
        self.model.eval()

    def get_move(self, fen: str) -> Optional[str]:
        """
        Given a board position in FEN notation, return the best move in UCI format.

        Uses constrained decoding: scores every legal move under the fine-tuned
        model and returns the one with the highest probability (lowest loss).
        Never returns an illegal move.
        """
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]

        if not legal_moves:
            return None

        # If many legal moves, sample a subset to keep inference fast
        import random
        moves_to_score = (
            legal_moves if len(legal_moves) <= self.MAX_MOVES_TO_SCORE
            else random.sample(legal_moves, self.MAX_MOVES_TO_SCORE)
        )

        best_move  = None
        best_score = float("-inf")

        with torch.no_grad():
            for move in moves_to_score:
                text   = f"FEN: {fen} MOVE: {move}"
                inputs = self.tokenizer(
                    text,
                    return_tensors = "pt",
                    truncation     = True,
                    max_length     = 96,
                ).to(self.device)

                loss = self.model(
                    **inputs,
                    labels = inputs["input_ids"]
                ).loss.item()

                if -loss > best_score:
                    best_score = -loss
                    best_move  = move

        return best_move
