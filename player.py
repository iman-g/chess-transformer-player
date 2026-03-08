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
