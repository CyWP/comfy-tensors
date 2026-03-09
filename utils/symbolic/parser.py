import re
import math
import torch


# ----------------------------
# AST nodes with tensor-aware repr
# ----------------------------
class Number:
    def __init__(self, v):
        # preserve int if possible
        if isinstance(v, float) and v.is_integer():
            self.v = int(v)
        else:
            self.v = v

    def __repr__(self):
        if isinstance(self.v, torch.Tensor):
            return f"Number(tensor shape={tuple(self.v.shape)})"
        return f"Number({self.v})"


class Variable:
    def __init__(self, n, value=None):
        self.n = n
        self.value = value

    def __repr__(self):
        if isinstance(self.value, torch.Tensor):
            return f"Variable({self.n}, tensor shape={tuple(self.value.shape)})"
        elif self.value is not None:
            return f"Variable({self.n}, {self.value})"
        else:
            return f"Variable({self.n})"


class BinOp:
    def __init__(self, l, op, r):
        self.l = l
        self.op = op
        self.r = r

    def __repr__(self):
        return f"BinOp({self.l}, '{self.op}', {self.r})"


class UnaryOp:
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def __repr__(self):
        return f"UnaryOp('{self.op}', {self.operand})"


class Func:
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"Func('{self.name}', {self.args})"


# ----------------------------
# Base Parser
# ----------------------------
class Parser:
    TOKEN_RE = re.compile(
        r"\s*(?:(\d+(\.\d*)?)|([A-Za-z_][A-Za-z_0-9]*)|(\*\*|==|!=|<=|>=|[(),+\-*/^~:&|<>:]))"
    )

    def __init__(self, functions=None, operators=None, constants=None):
        self.functions = {}
        self.operators = {
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
        self.constants = {"pi": math.pi, "e": math.e, "eps": 1e-8}

        if functions:
            self.functions.update(functions)
        if operators:
            self.operators.update(operators)
        if constants:
            self.constants.update(constants)

    # ----------------------------
    # idx helper
    # ----------------------------
    def idx(self, tensor, *args):
        slices = []
        for a in args:
            if isinstance(a, slice):
                slices.append(a)
            elif isinstance(a, (int, float)):
                slices.append(int(a))
            else:
                raise TypeError(f"Unsupported index type: {type(a)}")
        return tensor[tuple(slices)]

    # ----------------------------
    # Tokenizer
    # ----------------------------
    def tokenize(self, expr):
        for number, _, name, op in self.TOKEN_RE.findall(expr):
            if number:
                # preserve int if possible
                if "." in number:
                    yield ("NUMBER", float(number))
                else:
                    yield ("NUMBER", int(number))
            elif name:
                yield ("NAME", name)
            else:
                yield ("OP", op)

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else ("EOF", None)

    def consume(self):
        tok = self.peek()
        self.pos += 1
        return tok

    # ----------------------------
    # Recursive descent parser (precedence handled)
    # ----------------------------
    def parse(self, expr):
        self.tokens = list(self.tokenize(expr))
        self.pos = 0
        return self.expr()

    def expr(self):  # lowest precedence
        node = self.logic_or()
        return node

    def logic_or(self):
        node = self.logic_and()
        while self.peek()[1] == "|":
            op = self.consume()[1]
            node = BinOp(node, op, self.logic_and())
        return node

    def logic_and(self):
        node = self.comparison()
        while self.peek()[1] == "&":
            op = self.consume()[1]
            node = BinOp(node, op, self.comparison())
        return node

    def comparison(self):
        node = self.additive()
        while self.peek()[1] in ("==", "!=", "<", "<=", ">", ">="):
            op = self.consume()[1]
            node = BinOp(node, op, self.additive())
        return node

    def additive(self):
        node = self.multiplicative()
        while self.peek()[1] in ("+", "-"):
            op = self.consume()[1]
            node = BinOp(node, op, self.multiplicative())
        return node

    def multiplicative(self):
        node = self.power()
        while self.peek()[1] in ("*", "/", "//", "%", "@"):
            op = self.consume()[1]
            node = BinOp(node, op, self.power())
        return node

    def power(self):
        node = self.unary()
        if self.peek()[1] == "**":
            self.consume()
            node = BinOp(node, "**", self.power())
        return node

    def unary(self):
        if self.peek()[1] in ("+", "-", "neg", "not"):
            op = self.consume()[1]
            return UnaryOp(op, self.unary())
        return self.atom()

    def atom(self):
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

    def arg_list(self):
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

    def eval(self, node, vars={}):
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
            # Only propagate the current node's error
            raise RuntimeError(
                f"Error evaluating node {type(node).__name__} ({getattr(node, 'op', getattr(node, 'name', ''))}): {e}"
            ) from e

    def compute(self, expr, vars={}):
        try:
            ast = self.parse(expr)
            return self.eval(ast, vars)
        except Exception as e:
            raise RuntimeError(f"Error evaluating expression '{expr}': {e}") from e
