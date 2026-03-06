# player.py
import chess
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import Optional

try:
    # When running inside the chess_exam tournament
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

    def __init__(self, name: str = "TransformerPlayer"):
        super().__init__(name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        HF_MODEL    = "Iman-ghotbi/chess-gpt2-finetuned"

        print(f"Loading model from {HF_MODEL} on {self.device}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(HF_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(HF_MODEL)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded ✓")

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]

        if not legal_moves:
            return None

        # Cap moves to score for speed (still never returns illegal move)
        import random
        moves_to_score = legal_moves
        if len(legal_moves) > 30:
            moves_to_score = random.sample(legal_moves, 30)

        best_move, best_score = None, float('-inf')

        with torch.no_grad():
            for move in moves_to_score:
                text   = f"FEN: {fen} MOVE: {move}"
                inputs = self.tokenizer(
                    text,
                    return_tensors = 'pt',
                    truncation     = True,
                    max_length     = 96
                ).to(self.device)

                loss = self.model(
                    **inputs,
                    labels = inputs['input_ids']
                ).loss.item()

                if -loss > best_score:
                    best_score = -loss
                    best_move  = move

        return best_move
