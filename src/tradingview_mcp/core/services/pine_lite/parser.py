"""
pine_lite.parser — recursive-descent parser producing a small AST for the
supported Pine subset.

Grammar:
    statement  := multi_assign | assign | funcdef | call_stmt
    multi_assign := '[' IDENT (',' IDENT)* ']' '=' expr    # ta.macd only, enforced by validator
    assign     := IDENT '=' expr
    funcdef    := IDENT '(' (IDENT (',' IDENT)*)? ')' '=>' expr   # single-expression body only
    call_stmt  := call                                      # bare call, e.g. plot(x)

    expr       := ternary
    ternary    := or_expr ('?' expr ':' expr)?
    or_expr    := and_expr ('or' and_expr)*
    and_expr   := not_expr ('and' not_expr)*
    not_expr   := 'not' not_expr | comparison
    comparison := additive (('>'|'<'|'>='|'<='|'=='|'!=') additive)*
    additive   := term (('+'|'-') term)*
    term       := unary (('*'|'/'|'%') unary)*
    unary      := ('-'|'+') unary | postfix
    postfix    := primary ('[' NUM ']')*                    # historical reference x[1]
    primary    := NUM | STR | 'true' | 'false' | dotted_ident (call_args)? | '(' expr ')'
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union

from .tokenizer import tokenize, Token, TokenizeError

# Pine allows an optional explicit type annotation before a declaration
# (`float rng = high - low`, `int len = 20`). pine_lite has no type system
# (everything is just a series/scalar), so the type keyword is stripped —
# it carries no information the evaluator needs.
_TYPE_PREFIX_RE = re.compile(r"^(float|int|bool|string|color)\s+(?=[A-Za-z_]\w*\s*(=|\())")

# Reserved Pine keywords this subset doesn't support — matched against the
# first word of a [SCRIPT] line so the error message is immediately
# actionable instead of a generic syntax error from the tokenizer/parser.
_RESERVED_UNSUPPORTED = {
    "var": "persistent state (var/varip) is not supported by pine_lite v1 — recompute the value each bar instead",
    "varip": "persistent state (var/varip) is not supported by pine_lite v1",
    "for": "for-loops are not supported by pine_lite v1 — express the logic with ta.highest/ta.lowest/ta.sma-style built-ins instead",
    "while": "while-loops are not supported by pine_lite v1",
    "type": "user-defined types (type ...) are not supported by pine_lite v1",
    "import": "library imports are not supported by pine_lite v1 — inline the logic directly",
    "switch": "switch expressions are not supported by pine_lite v1 — use nested ternaries (cond ? a : b) instead",
    "if": "multi-line if/else blocks are not supported by pine_lite v1 — rewrite as a ternary: cond ? a : b",
}


class ParseError(Exception):
    def __init__(self, message: str, pos: int):
        super().__init__(message)
        self.message = message
        self.pos = pos


@dataclass
class Num:
    value: float


@dataclass
class Str:
    value: str


@dataclass
class Var:
    name: str


@dataclass
class Index:
    base: "Node"
    offset: int


@dataclass
class Call:
    name: str
    args: list
    kwargs: dict


@dataclass
class BinOp:
    op: str
    left: "Node"
    right: "Node"


@dataclass
class UnaryOp:
    op: str
    operand: "Node"


@dataclass
class Ternary:
    cond: "Node"
    then: "Node"
    otherwise: "Node"


@dataclass
class Assign:
    name: str
    expr: "Node"


@dataclass
class MultiAssign:
    names: list
    expr: "Node"


@dataclass
class FuncDef:
    name: str
    params: list
    body: "Node"


@dataclass
class CallStmt:
    call: Call


Node = Union[Num, Str, Var, Index, Call, BinOp, UnaryOp, Ternary]
Statement = Union[Assign, MultiAssign, FuncDef, CallStmt]


class _Parser:
    def __init__(self, tokens: list[Token]):
        self.toks = tokens
        self.i = 0

    def peek(self) -> Token:
        return self.toks[self.i]

    def peek_at(self, offset: int) -> Token:
        j = self.i + offset
        return self.toks[j] if j < len(self.toks) else self.toks[-1]

    def advance(self) -> Token:
        t = self.toks[self.i]
        self.i += 1
        return t

    def expect_op(self, op: str) -> Token:
        t = self.peek()
        if t.kind != "OP" or t.value != op:
            raise ParseError(f"expected '{op}' but found '{t.value or t.kind}'", t.pos)
        return self.advance()

    def at_end(self) -> bool:
        return self.peek().kind == "EOF"

    def _expect_end(self):
        if not self.at_end():
            t = self.peek()
            raise ParseError(f"unexpected trailing token '{t.value or t.kind}'", t.pos)

    def _expect_ident(self) -> str:
        t = self.peek()
        if t.kind != "IDENT":
            raise ParseError(f"expected identifier, found '{t.value or t.kind}'", t.pos)
        self.advance()
        return t.value

    # ---- statement-level ----

    def parse_statement(self) -> Statement:
        if self.peek().kind == "OP" and self.peek().value == "[":
            self.advance()
            names = [self._expect_ident()]
            while self.peek().kind == "OP" and self.peek().value == ",":
                self.advance()
                names.append(self._expect_ident())
            self.expect_op("]")
            self.expect_op("=")
            expr = self.parse_expr()
            self._expect_end()
            return MultiAssign(names, expr)

        start = self.i
        if self.peek().kind == "IDENT":
            name_tok = self.peek()
            nxt = self.peek_at(1)
            if nxt.kind == "OP" and nxt.value == "=":
                self.advance()
                self.advance()
                expr = self.parse_expr()
                self._expect_end()
                return Assign(name_tok.value, expr)
            if nxt.kind == "OP" and nxt.value == "(":
                save = self.i
                try:
                    self.advance()  # name
                    self.advance()  # (
                    params = []
                    if not (self.peek().kind == "OP" and self.peek().value == ")"):
                        params.append(self._expect_ident())
                        while self.peek().kind == "OP" and self.peek().value == ",":
                            self.advance()
                            params.append(self._expect_ident())
                    self.expect_op(")")
                    if self.peek().kind == "OP" and self.peek().value == "=>":
                        self.advance()
                        body = self.parse_expr()
                        self._expect_end()
                        return FuncDef(name_tok.value, params, body)
                except ParseError:
                    pass
                self.i = save  # not a funcdef -- fall through to a plain expression/call

        expr = self.parse_expr()
        self._expect_end()
        if isinstance(expr, Call):
            return CallStmt(expr)
        raise ParseError("expected an assignment, function definition, or function call statement", start)

    # ---- expression-level (precedence climbing) ----

    def parse_expr(self) -> Node:
        return self._ternary()

    def _ternary(self) -> Node:
        cond = self._or()
        if self.peek().kind == "OP" and self.peek().value == "?":
            self.advance()
            then = self.parse_expr()
            self.expect_op(":")
            otherwise = self.parse_expr()
            return Ternary(cond, then, otherwise)
        return cond

    def _or(self) -> Node:
        left = self._and()
        while self.peek().kind == "IDENT" and self.peek().value == "or":
            self.advance()
            right = self._and()
            left = BinOp("or", left, right)
        return left

    def _and(self) -> Node:
        left = self._not()
        while self.peek().kind == "IDENT" and self.peek().value == "and":
            self.advance()
            right = self._not()
            left = BinOp("and", left, right)
        return left

    def _not(self) -> Node:
        if self.peek().kind == "IDENT" and self.peek().value == "not":
            self.advance()
            return UnaryOp("not", self._not())
        return self._comparison()

    def _comparison(self) -> Node:
        left = self._additive()
        while self.peek().kind == "OP" and self.peek().value in (">", "<", ">=", "<=", "==", "!="):
            op = self.advance().value
            right = self._additive()
            left = BinOp(op, left, right)
        return left

    def _additive(self) -> Node:
        left = self._term()
        while self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            op = self.advance().value
            right = self._term()
            left = BinOp(op, left, right)
        return left

    def _term(self) -> Node:
        left = self._unary()
        while self.peek().kind == "OP" and self.peek().value in ("*", "/", "%"):
            op = self.advance().value
            right = self._unary()
            left = BinOp(op, left, right)
        return left

    def _unary(self) -> Node:
        if self.peek().kind == "OP" and self.peek().value in ("-", "+"):
            op = self.advance().value
            return UnaryOp(op, self._unary())
        return self._postfix()

    def _postfix(self) -> Node:
        node = self._primary()
        while self.peek().kind == "OP" and self.peek().value == "[":
            self.advance()
            idx_tok = self.peek()
            if idx_tok.kind != "NUM":
                raise ParseError("expected an integer inside [] (historical reference)", idx_tok.pos)
            self.advance()
            self.expect_op("]")
            node = Index(node, int(float(idx_tok.value)))
        return node

    def _primary(self) -> Node:
        t = self.peek()
        if t.kind == "NUM":
            self.advance()
            return Num(float(t.value))
        if t.kind == "STR":
            self.advance()
            return Str(t.value)
        if t.kind == "OP" and t.value == "(":
            self.advance()
            e = self.parse_expr()
            self.expect_op(")")
            return e
        if t.kind == "IDENT":
            name = self._dotted_ident()
            if self.peek().kind == "OP" and self.peek().value == "(":
                return self._call_args(name)
            if name == "true":
                return Num(1.0)
            if name == "false":
                return Num(0.0)
            return Var(name)
        raise ParseError(f"unexpected token '{t.value or t.kind}'", t.pos)

    def _dotted_ident(self) -> str:
        parts = [self._expect_ident()]
        while (
            self.peek().kind == "OP"
            and self.peek().value == "."
            and self.peek_at(1).kind == "IDENT"
        ):
            self.advance()
            parts.append(self._expect_ident())
        return ".".join(parts)

    def _call_args(self, name: str) -> Call:
        self.expect_op("(")
        args, kwargs = [], {}
        if not (self.peek().kind == "OP" and self.peek().value == ")"):
            self._parse_one_arg(args, kwargs)
            while self.peek().kind == "OP" and self.peek().value == ",":
                self.advance()
                self._parse_one_arg(args, kwargs)
        self.expect_op(")")
        return Call(name, args, kwargs)

    def _parse_one_arg(self, args: list, kwargs: dict):
        if (
            self.peek().kind == "IDENT"
            and self.peek_at(1).kind == "OP"
            and self.peek_at(1).value == "="
        ):
            kw = self.advance().value
            self.advance()  # '='
            kwargs[kw] = self.parse_expr()
        else:
            args.append(self.parse_expr())


def parse_statement_line(line: str, line_no: int):
    """Parse one [SCRIPT] line. Returns (Statement, None), (None, None) for
    a blank/comment-only line, or (None, issue_dict) on error."""
    stripped = line.split("//", 1)[0].strip()
    if not stripped:
        return None, None
    stripped = _TYPE_PREFIX_RE.sub("", stripped)
    first_word = stripped.split()[0].split("(")[0]
    if first_word in _RESERVED_UNSUPPORTED:
        return None, {
            "line": line_no, "text": stripped, "construct": first_word,
            "message": _RESERVED_UNSUPPORTED[first_word],
        }
    try:
        toks = tokenize(stripped)
        return _Parser(toks).parse_statement(), None
    except (TokenizeError, ParseError) as e:
        return None, {"line": line_no, "text": stripped, "message": e.message}


def parse_expression_text(text: str, line_no: int):
    """Parse a condition-block body as ONE expression. Multiple physical
    lines (a wrapped expression) are joined with spaces first. Returns
    (Node, None), (None, None) for an empty block, or (None, issue_dict)."""
    joined = " ".join(
        ln.split("//", 1)[0].strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("//")
    )
    if not joined:
        return None, None
    try:
        toks = tokenize(joined)
        p = _Parser(toks)
        expr = p.parse_expr()
        if not p.at_end():
            t = p.peek()
            raise ParseError(f"unexpected trailing token '{t.value or t.kind}'", t.pos)
        return expr, None
    except (TokenizeError, ParseError) as e:
        return None, {"line": line_no, "text": joined, "message": e.message}
