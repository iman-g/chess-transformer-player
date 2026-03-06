# Chess Transformer Player
### INFOMTALC 2026 — Midterm Assignment 1

A transformer-based chess player built on fine-tuned GPT-2 medium.

---

## Approach

### Model
- **Base model**: GPT-2 medium (355M parameters, decoder-only transformer)
- **Fine-tuned on**: 350k chess positions formatted as `"FEN: <position> MOVE: <uci>"`
- **Hosted on**: [HuggingFace — Iman-ghotbi/chess-gpt2-finetuned](https://huggingface.co/Iman-ghotbi/chess-gpt2-finetuned)

### Training Data
| Source | Size | Quality |
|---|---|---|
| Lichess Elite Database (4 months, 2022–2024) | 8.7M positions (2500+ ELO, no bullet) | High |
| Stockfish depth-8 annotations | 200k positions | Highest |
| **Training subset** | **350k positions** | Mixed |

### Move Selection: Constrained Decoding
Instead of generating moves freely (which produces illegal moves), the player:
1. Gets all legal moves from `python-chess`
2. Scores each legal move: `score = -loss(model, "FEN: {fen} MOVE: {move}")`
3. Returns the move with the **highest probability** under the model

This guarantees **0 illegal moves** and **0 fallbacks** in every game.

### Training Details
- Loss masking: only computed on MOVE tokens, not FEN tokens
- Optimizer: AdamW fused, lr=5e-5, 3 epochs
- Hardware: Kaggle GPU P100
- Final training loss: ~0.83

---

## Repository Structure

```
chess-transformer-player/
├── player.py              ← TransformerPlayer class (submission file)
├── requirements.txt       ← Dependencies
├── 1_prepare_data.ipynb   ← Data download + preparation (run on Colab)
├── 2_train_model.ipynb    ← Model fine-tuning (run on Kaggle GPU)
└── README.md
```

## Usage

```python
from player import TransformerPlayer

player = TransformerPlayer("MyPlayer")
move   = player.get_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
print(move)  # e2e4
```

## Results vs RandomPlayer (10 games)
- **Wins**: 6
- **Draws**: 4  
- **Losses**: 0
- **Score**: 8.0/10
- **Fallbacks**: 0
