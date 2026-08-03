"""
pine_lite.tokenizer — lexer for the constrained Pine Script subset.

Recognizes numbers, quoted strings, identifiers, and the punctuation the
supported expression grammar uses (parens, brackets for historical-
reference indexing, commas, dot for namespace/member access, comparison/
arithmetic/assignment operators, the `=>` function-definition arrow).
Line comments (`//...`) are stripped. This is intentionally a narrow lexer
for the supported subset described in pine_lite/__init__.py, not a general
Pine lexer.
"""
from __future__ import annotations

from dataclasses import dataclass

_SYMBOLS_2 = {">=", "<=", "==", "!=", "=>"}
_SYMBOLS_1 = set("+-*/%()[],.?:=<>")


@dataclass
class Token:
    kind: str  # "NUM", "STR", "IDENT", "OP", "EOF"
    value: str
    pos: int


class TokenizeError(Exception):
    def __init__(self, message: str, pos: int):
        super().__init__(message)
        self.message = message
        self.pos = pos


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            break  # rest of the line is a comment
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            start = i
            seen_dot = False
            while i < n and (text[i].isdigit() or (text[i] == "." and not seen_dot)):
                if text[i] == ".":
                    seen_dot = True
                i += 1
            tokens.append(Token("NUM", text[start:i], start))
            continue
        if c in "\"'":
            quote = c
            start = i
            i += 1
            while i < n and text[i] != quote:
                i += 1
            if i >= n:
                raise TokenizeError(f"unterminated string literal starting at position {start}", start)
            i += 1
            tokens.append(Token("STR", text[start + 1 : i - 1], start))
            continue
        if c.isalpha() or c == "_":
            start = i
            while i < n and (text[i].isalnum() or text[i] == "_"):
                i += 1
            tokens.append(Token("IDENT", text[start:i], start))
            continue
        two = text[i : i + 2]
        if two in _SYMBOLS_2:
            tokens.append(Token("OP", two, i))
            i += 2
            continue
        if c in _SYMBOLS_1:
            tokens.append(Token("OP", c, i))
            i += 1
            continue
        raise TokenizeError(f"unexpected character {c!r} at position {i}", i)
    tokens.append(Token("EOF", "", n))
    return tokens
