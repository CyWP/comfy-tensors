"""Base recursive-descent expression parser.

Provides AST node classes and a generic parser supporting:
- Arithmetic, comparison, logical, and bitwise operators
- Functions with arbitrary arguments
- Constants (pi, e, eps)
- Tensor indexing via slice syntax
"""

import math
import re

import torch


class Number:
    """Literal numeric value (int or float).

    Attributes:
        v: The numeric value.
    """

    def __init__(self, v: int | float | torch.Tensor):
        if isinstance(v, float) and v.is_integer():
            self.v = int(v)
        else:
            self.v = v

    def __repr__(self):
        if isinstance(self.v, torch.Tensor):
            return f"Number(tensor shape={tuple(self.v.shape)})"
        return f"Number({self.v})"


class Variable:
    """Named variable reference.

    Attributes:
        n: Variable name.
        value: Resolved value (set during evaluation).
    """

    def __init__(self, n: str, value: int | float | torch.Tensor | None = None):
        self.n = n
        self.value = value

    def __repr__(self):
        if isinstance(self.value, torch.Tensor):
            return f"Variable({self.n}, tensor shape={tuple(self.value.shape)})"
        elif self.value is not None:
            return f"Variable({self.n}, {self.value})"
        return f"Variable({self.n})"


class BinOp:
    """Binary operation node.

    Attributes:
        l: Left operand AST node.
        op: Operator symbol string.
        r: Right operand AST node.
    """

    def __init__(self, l: Number | Variable | BinOp | UnaryOp | Func, op: str, r: Number | Variable | BinOp | UnaryOp | Func):
        self.l = l
        self.op = op
        self.r = r

    def __repr__(self):
        return f"BinOp({self.l}, '{self.op}', {self.r})"


class UnaryOp:
    """Unary operation node.

    Attributes:
        op: Operator symbol string.
        operand: Operand AST node.
    """

    def __init__(self, op: str, operand: Number | Variable | BinOp | UnaryOp | Func):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"UnaryOp('{self.op}', {self.operand})"


class Func:
    """Function call node.

    Attributes:
        name: Function name.
        args: List of argument AST nodes.
    """

    def __init__(self, name: str, args: list[Number | Variable | BinOp | UnaryOp | Func]):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"Func('{self.name}', {self.args})"


class Parser:
    """Base recursive-descent expression parser.

    Supports operators with precedence (lowest to highest):
    logic_or -> logic_and -> comparison -> additive -> multiplicative -> power -> unary -> atom

    Attributes:
        functions: Dict mapping function names to callables.
        operators: Dict mapping operator symbols to binary callables.
        constants: Dict mapping constant names to values.
    """

    TOKEN_RE = re.compile(
        r"\s*(?:(\d+(\.\d*)?)|([A-Za-z_][A-Za-z_0-9]*)|(\*\*|==|!=|<=|>=|[()),+\-*/^~:&|<>:]))"
    )

    def __init__(
        self,
        functions: dict[str, callable] | None = None,
        operators: dict[str, callable] | None = None,
        constants: dict[str, int | float] | None = None,
    ):
        self.functions: dict[str, callable] = {}
        self.operators: dict[str, callable] = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b,
            "//": lambda a, b: a // b,
            "**": lambda a, b: a**b,
            "%": lambda a, b: a % b,
            "@": lambda a, b: a @ b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "&": lambda a, b: a & b,
            "|": lambda a, b: a | b,
            "^": lambda a, b: a ^ b,
            "neg": lambda x: -x,
            "not": lambda x: ~x,
            ":": lambda a=None, b=None, c=None: slice(a, b, c),
        }
        self.constants: dict[str, float] = {"pi": math.pi, "e": math.e, "eps": 1e-8}

        if functions:
            self.functions.update(functions)
        if operators:
            self.operators.update(operators)
        if constants:
            self.constants.update(constants)

    def idx(self, tensor: torch.Tensor, *args: int | float | slice) -> torch.Tensor:
        """Indexes a tensor with support for slices.

        Args:
            tensor: Tensor to index.
            *args: Indices which may be int, float, or slice.

        Returns:
            Indexed tensor.

        Raises:
            TypeError: If index type is unsupported.
        """
        slices = []
        for a in args:
            if isinstance(a, slice):
                slices.append(a)
            elif isinstance(a, (int, float)):
                slices.append(int(a))
            else:
                raise TypeError(f"Unsupported index type: {type(a)}")
        return tensor[tuple(slices)]

    def tokenize(self, expr: str):
        """Tokenizes an expression string.

        Args:
            expr: Expression string to tokenize.

        Yields:
            Tuple of (token_type, token_value).
        """
        for number, _, name, op in self.TOKEN_RE.findall(expr):
            if number:
                if "." in number:
                    yield ("NUMBER", float(number))
                else:
                    yield ("NUMBER", int(number))
            elif name:
                yield ("NAME", name)
            else:
                yield ("OP", op)

    def peek(self):
        """Returns the next token without consuming it."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)

    def consume(self):
        """Consumes and returns the next token."""
        tok = self.peek()
        self.pos += 1
        return tok

    def parse(self, expr: str) -> Number | Variable | BinOp | UnaryOp | Func:
        """Parses an expression string into an AST.

        Args:
            expr: Expression string to parse.

        Returns:
            Root AST node.
        """
        self.tokens = list(self.tokenize(expr))
        self.pos = 0
        return self.expr()

    def expr(self) -> Number | Variable | BinOp | UnaryOp | Func:
        """Parses lowest-precedence expression (logic_or)."""
        node = self.logic_or()
        return node

    def logic_or(self) -> Number | Variable | BinOp | UnaryOp | Func:
        """Parses logic_or expressions (lowest precedence)."""
        node = self.logic_and()
        while self.peek()[1] == "|":
            op = self.consume()[1]
            node = BinOp(node, op, self.logic_and())
        return node

    def logic_and(self) -> Number | Variable | BinOp | UnaryOp | Func:
        """Parses logic_and expressions."""
        node = self.comparison()
        while self.peek()[1] == "&":
            op = self.consume()[1]
            node = BinOp(node, op, self.comparison())
        return node

    def comparison(self) -> Number | Variable | BinOp | UnaryOp | Func:
        """Parses comparison expressions."""
        node = self.additive()
        while self.peek()[1] in ("==", "!=", "<", "<=", ">", ">="):
            op = self.consume()[1]
            node = BinOp(node, op, self.additive())
        return node

    def additive(self) -> Number | Variable | BinOp | UnaryOp | Func:
        """Parses additive expressions (+, -)."""
        node = self.multiplicative()
        while self.peek()[1] in ("+", "-"):
            op = self.consume()[1]
            node = BinOp(node, op, self.multiplicative())
        return node

    def multiplicative(self) -> Number | Variable | BinOp | UnaryOp | Func:
        """Parses multiplicative expressions (*, /, //, %, @)."""
        node = self.power()
        while self.peek()[1] in ("*", "/", "//", "%", "@"):
            op = self.consume()[1]
            node = BinOp(node, op, self.power())
        return node

    def power(self) -> Number | Variable | BinOp | UnaryOp | Func:
        """Parses power expressions (**)."""
        node = self.unary()
        if self.peek()[1] == "**":
            self.consume()
            node = BinOp(node, "**", self.power())
        return node

    def unary(self) -> Number | Variable | BinOp | UnaryOp | Func:
        """Parses unary expressions (+, -, neg, not)."""
        if self.peek()[1] in ("+", "-", "neg", "not"):
            op = self.consume()[1]
            return UnaryOp(op, self.unary())
        return self.atom()

    def atom(self) -> Number | Variable | BinOp | UnaryOp | Func:
        """Parses atomic expressions (numbers, variables, parenthesized exprs)."""
        tok_type, tok_val = self.consume()
        if tok_type == "NUMBER":
            return Number(tok_val)
        if tok_val == "(":
            node = self.expr()
            if self.consume()[1] != ")":
                raise SyntaxError("Expected ')'")
            return node
        if tok_type == "NAME":
            name = tok_val
            if self.peek()[1] == "(":
                self.consume()
                args = self.arg_list()
                return Func(name, args)
            if name in self.functions:
                return Func(name, [self.atom()])
            return Variable(name)
        if tok_val == ":":
            return slice(None)
        raise SyntaxError(f"Unexpected token: {tok_val}")

    def arg_list(self) -> list[Number | Variable | BinOp | UnaryOp | Func]:
        """Parses a function argument list."""
        args = []
        if self.peek()[1] == ")":
            self.consume()
            return args
        args.append(self.expr())
        while self.peek()[1] == ",":
            self.consume()
            args.append(self.expr())
        if self.consume()[1] != ")":
            raise SyntaxError("Expected ')' after arguments")
        return args

    def eval(
        self,
        node: Number | Variable | BinOp | UnaryOp | Func,
        vars: dict[str, int | float | torch.Tensor] | None = None,
    ) -> int | float | torch.Tensor:
        """Evaluates an AST node.

        Args:
            node: AST node to evaluate.
            vars: Dict of variable names to values.

        Returns:
            Evaluated result.

        Raises:
            RuntimeError: If variable or function is unknown or evaluation fails.
        """
        if vars is None:
            vars = {}

        try:
            if isinstance(node, Number):
                return node.v
            if isinstance(node, Variable):
                val = vars.get(node.n, self.constants.get(node.n))
                if val is None:
                    raise RuntimeError(f"Unknown variable '{node.n}'")
                node.value = val
                return val
            if isinstance(node, UnaryOp):
                operand = self.eval(node.operand, vars)
                try:
                    return self.operators[node.op](operand)
                except Exception as e:
                    shape_info = (
                        f"tensor shape={tuple(operand.shape)}"
                        if isinstance(operand, torch.Tensor)
                        else repr(operand)
                    )
                    raise RuntimeError(
                        f"Error in unary operator '{node.op}' with operand {shape_info}: {e}"
                    ) from e
            if isinstance(node, BinOp):
                l = self.eval(node.l, vars)
                r = self.eval(node.r, vars)
                try:
                    return self.operators[node.op](l, r)
                except Exception as e:
                    l_info = (
                        f"tensor shape={tuple(l.shape)}"
                        if isinstance(l, torch.Tensor)
                        else repr(l)
                    )
                    r_info = (
                        f"tensor shape={tuple(r.shape)}"
                        if isinstance(r, torch.Tensor)
                        else repr(r)
                    )
                    raise RuntimeError(
                        f"Error in operator '{node.op}' with operands {l_info} and {r_info}: {e}"
                    ) from e
            if isinstance(node, Func):
                fn = self.functions.get(node.name)
                if fn is None:
                    raise RuntimeError(f"Unknown function '{node.name}'")
                args = [self.eval(a, vars) for a in node.args]
                try:
                    return fn(*args)
                except Exception as e:
                    args_info = []
                    for a in args:
                        if isinstance(a, torch.Tensor):
                            args_info.append(f"tensor shape={tuple(a.shape)}")
                        else:
                            args_info.append(repr(a))
                    raise RuntimeError(
                        f"Error in function '{node.name}' with args {args_info}: {e}"
                    ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error evaluating node {type(node).__name__} "
                f"({getattr(node, 'op', getattr(node, 'name', ''))}): {e}"
            ) from e

    def compute(
        self, expr: str, vars: dict[str, int | float | torch.Tensor] | None = None
    ) -> int | float | torch.Tensor:
        """Parses and evaluates an expression.

        Args:
            expr: Expression string to evaluate.
            vars: Dict of variable names to values.

        Returns:
            Evaluated result.

        Raises:
            RuntimeError: If parsing or evaluation fails.
        """
        try:
            ast = self.parse(expr)
            return self.eval(ast, vars)
        except Exception as e:
            raise RuntimeError(f"Error evaluating expression '{expr}': {e}") from e
