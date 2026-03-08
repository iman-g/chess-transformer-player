import chess
import torch
import torch.nn.functional as F
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


# Opening book: hardcoded best moves for common positions
# Instant lookup, no model inference needed for opening moves
OPENING_BOOK = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1":       "e2e4",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1":    "e7e5",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2":  "g1f3",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": "b8c6",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1":    "d7d5",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2":  "c2c4",
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1":    "e7e5",
}


class TransformerPlayer(Player):
    """
    Chess player based on fine-tuned GPT-2 medium.

    Architecture: Decoder-only transformer (GPT-2 medium, 355M params)
    Training data: 350k positions from Lichess Elite Database (2500+ ELO)
                   + 200k positions re-annotated with local Stockfish (depth=8)
    Move selection: Batched constrained decoding — scores ALL legal moves in
                    a single forward pass, picks highest probability.
                    Guarantees 0 illegal moves and 0 fallbacks.
    """

    HF_MODEL   = "Iman-ghotbi/chess-gpt2-finetuned"
    BATCH_SIZE = 32   # moves scored per forward pass

    def __init__(self, name: str = "TransformerPlayer"):
        super().__init__(name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.HF_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(self.HF_MODEL)
        self.model.to(self.device)
        self.model.eval()

    def _score_moves_batched(self, fen: str, moves: list) -> list:
        """
        Score a list of moves in batches.
        Returns a list of scores (higher = better move).
        """
        all_scores = []

        for i in range(0, len(moves), self.BATCH_SIZE):
            batch_moves = moves[i : i + self.BATCH_SIZE]
            texts = [f"FEN: {fen} MOVE: {move}" for move in batch_moves]

            inputs = self.tokenizer(
                texts,
                return_tensors = "pt",
                truncation     = True,
                max_length     = 96,
                padding        = True,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits  # [batch, seq_len, vocab]

            # Per-sample loss (model.loss only returns batch mean)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs["input_ids"][:, 1:].contiguous()

            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction = "none",
            ).view(shift_labels.size())

            # Mask padding so it doesn't distort the score
            mask           = (shift_labels != self.tokenizer.pad_token_id).float()
            per_sample_loss = (loss_per_token * mask).sum(1) / mask.sum(1).clamp(min=1)

            all_scores.extend((-per_sample_loss).tolist())

        return all_scores
        
        
    def get_move(self, fen: str) -> Optional[str]:
        """
        Returns the best legal move, avoiding 3-fold repetition draws.
        """
        if fen in OPENING_BOOK:
            return OPENING_BOOK[fen]
    
        board       = chess.Board(fen)
        legal_moves = [m.uci() for m in board.legal_moves]
    
        if not legal_moves:
            return None
    
        # Score all moves in batches
        scores       = self._score_moves_batched(fen, legal_moves)
        scored_moves = sorted(zip(legal_moves, scores), key=lambda x: -x[1])
    
        # Pick best move that doesn't cause 3-fold repetition
        for move_uci, score in scored_moves:
            try:
                move = chess.Move.from_uci(move_uci)
                board.push(move)
                is_repetition = board.is_repetition(3)
                board.pop()
                if not is_repetition:
                    return move_uci
            except Exception:
                continue
    
        # All moves cause repetition (very rare) → just return best move
        return scored_moves[0][0]
