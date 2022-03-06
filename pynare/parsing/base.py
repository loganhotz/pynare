"""
module with keywords & key symbols, and validators used throughout
pynare Parser
"""

from typing import Union
from copy import deepcopy

# use autograd's thinly wrapped numpy & scipy to handle ArrayBoxes when 
#    evaluating derivatives
import autograd.numpy as np
from autograd.scipy.special import erf

import pynare.parsing.ast as ast



EOF      = 'EOF'
ID       = 'ID'
NUMBER   = 'NUMBER'
STRING   = 'STRING'
FUNCTION = 'FUNCTION'


COMMA       = ','
PERIOD      = '.'
SEMI        = ';'
COLON       = ':'
POUND       = '#'
AMPERSAND   = '&'
PIPE        = '|'
ATSIGN      = '@'
EXCLAMATION = '!'
TILDE       = '~'
PERCENT     = '%'
BACKSLASH   = '\\'

LT          = '<'
LTOE        = '<='
GT          = '>'
GTOE        = '>='
EQUALITY    = '=='

LPARE       = '('
RPARE       = ')'
LBRACKET    = '['
RBRACKET    = ']'
LBRACE      = '{'
RBRACE      = '}'
SQUOTE      = '\''
DQUOTE      = '\"'

PLUS    = 'PLUS'
MINUS   = 'MINUS'
MUL     = 'MUL'
DIV     = 'DIV'
POWER   = 'POWER'
EQUALS  = 'EQUALS'


# reserved function names
EXP     = 'exp'
LOG     = 'log'
LN      = 'ln'
LOG10   = 'log10'
SQRT    = 'sqrt'
ABS     = 'abs'
SIGN    = 'sign'
SIN     = 'sin'
COS     = 'cos'
TAN     = 'tan'
ASIN    = 'asin'
ACOS    = 'acos'
ATAN    = 'atan'
MAX     = 'max'
MIN     = 'min'
NORMCDF = 'normcdf'
NORMPDF = 'normpdf'
ERF     = 'erf'


class Token(object):

    def __init__(
        self,
        token_type: str,
        value: Union[str, float],
        line_number: int = None,
        line_pos: int = None
    ):
        self.type = token_type
        self.value = value
        self.line_number = line_number
        self.line_pos = line_pos


    def assign_loc(self, line_number, line_pos):
        self.line_number = line_number
        self.line_pos = line_pos

    def __str__(self):
        return (
            f"Token[{self.type}, {self.value}, {self.line_number}, {self.line_pos}]"
        )

    def __repr__(self):
        return f"Token[{self.type}, {repr(self.value)}]"


reserved_funcs = {
    EXP:        Token(FUNCTION, EXP),
    LOG:        Token(FUNCTION, LOG),
    LN:         Token(FUNCTION, LN),
    LOG10:      Token(FUNCTION, LOG10),
    SQRT:       Token(FUNCTION, SQRT),
    ABS:        Token(FUNCTION, ABS),
    SIGN:       Token(FUNCTION, SIGN),
    SIN:        Token(FUNCTION, SIN),
    COS:        Token(FUNCTION, COS),
    TAN:        Token(FUNCTION, TAN),
    ASIN:       Token(FUNCTION, ASIN),
    ACOS:       Token(FUNCTION, ACOS),
    ATAN:       Token(FUNCTION, ATAN),
    MAX:        Token(FUNCTION, MAX),
    MIN:        Token(FUNCTION, MIN),
    NORMCDF:    Token(FUNCTION, NORMCDF),
    NORMPDF:    Token(FUNCTION, NORMPDF),
    ERF:        Token(FUNCTION, ERF)
}


class ABCVisitor(object):
    """ Abstract Base Class for visit Abstract Syntax Tree Visitors """

    def __init__(
        self,
        parser
    ):
        try:
            self.tree = parser.parse()
        except AttributeError:
            self.tree = deepcopy(parser)

    def visit(self, node):
        # using the AST node's name to visit the appropriate method
        method = 'visit_' + type(node).__name__
        visitor = getattr(self, method, self._nonexistent_node)
        return visitor(node)

    def _nonexistent_node(self, node):
        raise Exception(f'No visit_{type(node).__name__} method')


class BaseEvaluator(object):

    def __new__(
        cls,
        tree,
        scope
    ):
        evaluator = _BaseEvaluator(tree, scope)
        return evaluator.evaluate()


class _BaseEvaluator(ABCVisitor):

    def __init__(
        self,
        tree,
        scope
    ):
        super().__init__(tree)
        self.scope = scope

    def evaluate(self):
        return self.visit(self.tree)

    def visit_BinaryOp(self, node):
        if node.op.type == PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == MUL:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == DIV:
            return self.visit(node.left) / self.visit(node.right)
        elif node.op.type == POWER:
            return self.visit(node.left) ** self.visit(node.right)

    def visit_CompOp(self, node):
        if node.op.type == LT:
            return self.visit(node.left) < self.visit(node.right)
        elif node.op.type == LTOE:
            return self.visit(node.left) <= self.visit(node.right)
        elif node.op.type == GT:
            return self.visit(node.left) > self.visit(node.right)
        elif node.op.type == GTOE:
            return self.visit(node.left) >= self.visit(node.right)
        elif node.op.type == EQUALITY:
            return self.visit(node.left) == self.visit(node.right)

    def visit_UnaryOp(self, node):
        if node.op.type == PLUS:
            return + self.visit(node.expr)
        elif node.op.type == MINUS:
            return - self.visit(node.expr)

    def visit_Num(self, node):
        return node.value

    def visit_Var(self, node):
        try:
            try:
                return self.scope.lookup(node.value)
            except AttributeError:
                return self.scope[node.value]

        except KeyError:
            raise SyntaxError(f"symbol '{node.value}' was not assigned value before use")

    def visit_Function(self, node):
        f = node.value

        if f == EXP:
            return np.exp(self.visit(node.expr))

        if (f == LOG) or (f == LN):
            return np.log(self.visit(node.expr))

        if f == LOG10:
            return np.log10(self.visit(node.expr))

        if f == SQRT:
            return np.sqrt(self.visit(node.expr))

        if f == ABS:
            return np.abs(self.visit(node.expr))

        if f == SIGN:
            return np.sign(self.visit(node.expr))

        if f == SIN:
            return np.sin(self.visit(node.expr))

        if f == COS:
            return np.cos(self.visit(node.expr))

        if f == TAN:
            return np.tan(self.visit(node.expr))

        if f == ASIN:
            return np.arcsin(self.visit(node.expr))

        if f == ACOS:
            return np.arccos(self.visit(node.expr))

        if f == ATAN:
            return np.arctan(self.visit(node.expr))

        if f == MAX:
            raise NotImplementedError(MAX)

        if f == MIN:
            raise NotImplementedError(MIN)

        if f == NORMCDF:
            raise NotImplementedError(NORMCDF)

        if f == NORMPDF:
            raise NotImplementedError(NORMPDF)

        if f == ERF:
            return erf(self.visit(node.expr))


class ASTSubstitution(object):

    def __new__(
        cls,
        tree,
        scope
    ):
        substitutor = _ASTSubstitution(tree, scope)
        return substitutor.simplify()


class _ASTSubstitution(ABCVisitor):

    def __init__(
        self,
        tree,
        scope
    ):
        super().__init__(tree)
        self.scope = scope

    def simplify(self):
        self.visit(self.tree)
        return self.tree

    def maybe_substitute(self, node, attr):
        attribute = getattr(node, attr)
        if isinstance(attribute, ast.Var):
            try:
                var_value = self.scope.lookup(attribute.value)
            except AttributeError:
                var_value = self.scope.get(attribute.value)

            if var_value is not None:
                setattr(node, attr, self.substitute(attribute, var_value))

    def substitute(self, node, var_value):
        var_name = node.value

        token = Token(NUMBER, var_value)
        return ast.Num(token)

    def visit_BinaryOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
        self.maybe_substitute(node, 'left')
        self.maybe_substitute(node, 'right')

    def visit_UnaryOp(self, node):
        self.visit(node.expr)
        self.maybe_substitute(node, 'expr')

    def visit_Num(self, node):
        pass

    def visit_Var(self, node):
        pass

    def visit_PeriodVar(self, node):
        pass

    def visit_Function(self, node):
        self.maybe_substitute(node, 'expr')
        self.visit(node.expr)
