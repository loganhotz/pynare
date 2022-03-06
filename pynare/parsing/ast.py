"""
module containing all the nodes and leaves of the Abstract Syntax Tree
generated by the pynare Parser
"""

from __future__ import annotations
from typing import List, Dict, Union, Any



class AST(object):
    """ Abstract Base Class for all nodes and leaves in AST """

    def __init__(self):
        pass

    def describe(self, level=0):

        indent = '\t'

        # gets all the non-dunder and non-function attributes of the subclass 
        #    into a list
        valid_attrs = [
            d for d in dir(self) if (d != 'describe')
            and not (d.startswith('__') and d.endswith('__'))
        ]

        # only prints the current AST object name when it's the first level
        if level == 0:
            print('{} with Attributes: '.format(self.__repr__()))

        for attr_name in valid_attrs:

            attr = getattr(self, attr_name)
            if isinstance(attr, ArgCompoundASTList):

                arg_list = [a for a in attr.arguments._list]
                print(f'{indent*(level+1)}{attr_name}{arg_list}')

                for single_ast in attr._list:
                    print(f'{indent*(level+2)}{single_ast}')

            elif isinstance(attr, CompoundASTList):

                print(f'{indent*(level+1)}{attr_name}')

                for single_ast in attr._list:
                    print(f'{indent*(level+2)}{single_ast}')

            elif isinstance(attr, type(self).__bases__):
                # if 'attr' is also a derived class of AST
                print(f'{indent*(level+1)}{attr_name} {getattr(self, attr_name)}')
                attr.describe(level=level+1)

    def __repr__(self):
        return f'<{type(self).__name__} Object>'


def _is_iterable_of_ast(obj):
    # returns True if obj is an iterable with all elements instances of AST
    if isinstance(obj, list):
        return all([isinstance(o, AST) for o in obj])
    elif isinstance(obj, dict):
        return all([_is_iterable_of_ast(v) for v in obj.values()])




# 
# Final AST structure for entire model file
#
class ModelFile(AST):

    def __init__(
        self,
        model_preamble: VariablePreamble,
        model_definition: ModelDefinition,
        model_actions: ModelActions
    ):
        self.model_preamble = model_preamble
        self.model_definition = model_definition
        self.model_actions = model_actions



#
# algebra and function nodes
#
class BinaryOp(AST):

    def __init__(
        self,
        left: AST,
        op: Token,
        right: AST
    ):
        self.left = left
        self.op = op
        self.right = right

    def __str__(self):
        return f'<BinaryOp[{self.op.value}]>'

class UnaryOp(AST):

    def __init__(
        self,
        op: Token,
        expr: AST
    ):
        self.op = op
        self.expr = expr

    def __str__(self):
        return f'<UnaryOp[{self.op.value}]>'

class CompOp(AST):

    def __init__(
        self,
        left: AST,
        op: Token,
        right: AST
    ):
        self.left = left
        self.op = op
        self.right = right

    def __str__(self):
        return f'<CompOp[{self.op.value}]>'

class Num(AST):

    def __init__(
        self,
        token: Token[NUMBER]
    ):
        self.token = token
        self.value = self.token.value

    def __str__(self):
        return f'<Num[{self.value}]>'

class Function(AST):

    def __init__(
        self,
        token: Token[FUNCTION],
        expr: AST
    ):
        self.token = token
        self.value = self.token.value
        self.expr = expr

class Var(AST):

    def __init__(
        self,
        token: Token[ID]
    ):
        self.token = token
        self.value = self.token.value

    def __str__(self):
        return f'<Var[{self.value}]>'

class PeriodVar(AST):

    def __init__(
        self,
        var: Var,
        direction: Union[Token[PLUS], Token[MINUS]],
        offset: Token[NUM]
    ):
        self.var = var
        self.value = self.var.value
        self.direction = direction
        self.offset = offset

        d, o = direction.value, offset.value
        self.period_offset = int(d + str(int(o)))

    def __str__(self):
        return f'<PeriodVar[{self.value} ({self.period_offset})]>'


#
# before model declaration, variables declared and values assigned
#
class VariablePreamble(AST):
    def __init__(
        self,
        declared_variables: CompoundASTList,
        assigned_parameters: CompoundASTList
    ):
        self.variables = declared_variables
        self.parameters = assigned_parameters


class VarDeclaration(AST):

    def __init__(
        self,
        var_node: Token,
        vtype: Token
    ):
        self.var_node = var_node
        self.vtype = vtype

    def __repr__(self):
        return f'<{self.__class__.__name__}[{self.var_node.value}, {self.vtype.value}]>'


class VarAssignment(AST):

    def __init__(
        self,
        left: Var,
        op: Token,
        right: AST
    ):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self):
        return f'<{self.__class__.__name__}[{self.left.value} = {self.right}]>'


#
# collections of AST nodes - Dict is not used yet
#
class CompoundASTDict(AST):

    def __init__(self):
        self._dict = dict()


class CompoundASTList(AST):

    def __init__(self):
        self._list = list()

    def append(self, item):
        self._list.append(item)

    def extend(self, item):
        self._list.extend(item)


class ArgCompoundASTList(CompoundASTList):

    def __init__(
        self,
        arguments: CompoundASTList
    ):
        self.arguments = arguments
        super().__init__()


#
# optional arguments for declaring different blocks - model, steady state, etc.
#
class Argument(AST):

    def __init__(
        self,
        token: Token,
        value: Union[Token[STRING], Token[ID], AST, Any] = None
    ):
        self.token = token
        self.value = value

    def __repr__(self):
        return f'<{self.__class__.__name__}[{self.token.value} = {self.value}]>'



#
# Defining the model
#
class ModelDefinition(AST):

    def __init__(
        self,
        model: Model,
        arguments: CompoundASTList
    ):
        self.model = model
        self.arguments = arguments

class Model(AST):

    def __init__(self):
        self._statements = CompoundASTList()

    def append_statement(self, item):
        self._statements.append(item)

class ModelExpression(AST):

    def __init__(
        self,
        left: AST,
        right: AST
    ):
        self.left = left
        self.right = right
        self.tag = None

    def add_tag(self, new_tag):
        self.tag = new_tag

class LocalModelDeclaration(VarAssignment):

    def __init__(
        self,
        left: AST,
        op: Token,
        right: AST
    ):
        super().__init__(left, op, right)



# 
# Boundary, simulation, and shocks commands that happen after model definition
#
class ModelActions(AST):

    def __init__(self):

        self.boundaries = ModelBoundaryValues()
        self.shocks = None

    def set_boundary_values(self, values, boundary):
        self.boundaries.set_boundary(values, boundary)

    def set_shocks(self, shocks):
        self.shocks = shocks


class ModelBoundaryValues(AST):

    def __init__(self):
        self.initval = None
        self.endval = None
        self.histval = None

    def set_boundary(self, values, boundary):
        setattr(self, boundary, values)


class ShocksBlock(ArgCompoundASTList):

    def __init__(
        self,
        arguments
    ):
        super().__init__(arguments)

class StochasticShock(AST):

    def __init__(self, shock_type, **kwargs):
        self.shock_type = shock_type
        self._print_attrs = dict()

        for k, v in kwargs.items():
            setattr(self, k, v)
            self._print_attrs[k] = v

    def __repr__(self):
        return f'<StochasticShock[{self.shock_type}, {self._print_attrs}]>'


class DeterministicShock(AST):

    def __init__(
        self,
        shock_type: str,
        shock: VAR,
        periods: CompoundASTList,
        values: CompoundASTList
    ):
        self.shock_type = shock_type
        self.shock = shock
        self.periods = periods
        self.values = values


class DeterministicShockPeriod(AST):
    def __init__(
        self,
        start: int,
        end: int
    ):
        self.start = start
        self.end = end


class DeterministicShockValue(AST):

    def __init__(
        self,
        value: AST
    ):
        self.value = value


class HomotopyVar(AST):

    def __init__(
        self,
        token: Token,
        end: AST,
        begin: AST = None
    ):
        self.token = token
        self.begin = begin
        self.end = end


#
# general parsing commands after model has been defined
#



