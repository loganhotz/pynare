"""
translating language-specific ASTs to a homogenous ModelOutline object that can
be interpreted by the top-level Model class, no matter the source language

The derived classes of ABCVisitor that walk the ASTs produce the final ModelOutline
object, and perform semantics checks while doing so
"""



from __future__ import annotations

import copy

from pynare.parsing.lexer import DynareLexer
from pynare.parsing.parser import DynareParser
from pynare.parsing.semantics import DynareSemanticAnalyzer

import pynare.parsing.ast as ast
import pynare.parsing.base as base
import pynare.parsing.dynare as dyn



class ModelOutline(object):

    def __init__(
        self,
        language: str
    ):
        self.language = language
        self._parameters = dict()
        self._endogenous = list()
        self._endo_lead_lags = dict()

        self._stochastic_exogenous = list()
        self._deterministic_exogenous = list()

        self._local_model_variables = dict()
        self._model_expression_asts = list()

        self._initial_values = dict()
        self._terminal_values = dict()
        self._historical_values = dict()

        self._shocks = {}


    def set_model_parameter(
        self,
        var_name: str,
        value: Union[int, float]
    ):
        self._parameters[var_name] = value


    def is_parameter(
        self,
        var_name: str
    ):
        return var_name in self._parameters


    def declare_variable(
        self,
        var_name: str,
        vtype: str
    ):
        if vtype == '_parameters':
            self._parameters[var_name] = None
        else:
            vtype_list = getattr(self, vtype)
            vtype_list.append(var_name)

    def init_lead_lag(
        self,
        var_name: str
    ):
        self._endo_lead_lags[var_name] = set()


    def add_model_expression_ast(
        self,
        mexpr_ast
    ):
        self._model_expression_asts.append(mexpr_ast)


    @property
    def model_scope(self):
        # the scope when walking the model definition ASTs and replacing 
        #    variables with parameters and local variables when appropriate
        return {**self._parameters, **self._local_model_variables}


    def __deepcopy__(self, memo=dict()):

        cls = self.__class__
        new_outline = cls.__new__(cls)
        memo[id(self)] = new_outline

        for k, v in self.__dict__.items():
            setattr(new_outline, k, copy.deepcopy(v, memo))

        super(ModelOutline, new_outline).__init__()
        return new_outline




class ScopedMemory(object):

    def __init__(
        self,
        enclosing_scope=None
    ):
        self._symbols = dict()
        self.enclosing_scope = enclosing_scope

    def insert(self, var_name, value):
        self._symbols[var_name] = value

    def lookup(self, var_name):

        value = self._symbols.get(var_name)
        if value is not None:
            return value

        if self.enclosing_scope is not None:
            return self.enclosing_scope.lookup(var_name)

        return None



class ModelFactory(object):

    def __new__(
        self,
        text: str,
        language: str
    ):
        if language == 'dynare':
            model_factory = DynareModelFactory(text)
            return model_factory.create_outline()

        else:
            raise NotImplementedError()



class DynareModelFactory(base.ABCVisitor):

    # mapping of dynare variable type names to the ModelOutline's internal
    #    variable names
    _vtype_mapping = {
        'parameters': '_parameters',
        'varexo': '_stochastic_exogenous',
        'varexo_det': '_deterministic_exogenous',
        'var': '_endogenous'
    }

    def __init__(
        self,
        text: str
    ):
        lexer = DynareLexer(text)
        parser = DynareParser(lexer)

        model_ast = parser.parse()
        semantics = DynareSemanticAnalyzer(model_ast)
        semantics.verify()

        super().__init__(model_ast)

        self.outline = ModelOutline('dynare')

        self.model_scope = ScopedMemory()
        self.file_scope = ScopedMemory(
            enclosing_scope=self.model_scope
        )


    def create_outline(self):
        self.visit(self.tree)
        return self.outline


    def visit_ModelFile(self, node):
        self.visit(node.model_preamble)
        self.visit(node.model_definition)
        self.visit(node.model_actions)


    #
    # Preamble
    #
    def visit_VariablePreamble(self, node):
        self.visit(node.variables)
        self.visit(node.parameters)


    def visit_VarDeclaration(self, node):

        var_name = node.var_node.value
        var_type = node.vtype.value
        self.outline.declare_variable(
            var_name=var_name,
            vtype=self._vtype_mapping[var_type]
        )

        if var_type in (
            dyn.VAR, dyn.PREDETERMINED, dyn.TREND_VAR, dyn.LOG_TREND_VAR
        ):
            self.outline.init_lead_lag(
                var_name=var_name
            )


    def visit_VarAssignment(self, node):
        """
        check if a variable was declared. if not, assume it's a local variable
        and save that to the scope and evaluate the value it's being assigned.
        If it was assumed, we assume it's a model parameter and assign the
        value to it in the ModelOutline.

        This function can handle all types of assignments 'a = b', but I really
        only intend this to be used for parameter & local variable assignments
        before the model definition
        """

        value = base.BaseEvaluator(
            tree=node.right,
            scope=self.file_scope
        )
        var_name = node.left.value

        if self.outline.is_parameter(var_name):
            self.model_scope.insert(
                var_name=var_name,
                value=value
            )
            self.outline.set_model_parameter(
                var_name=var_name,
                value=node.right
            )

        else:
            self.file_scope.insert(
                var_name=var_name,
                value=value
            )


    #
    # Model Block
    #
    def visit_ModelDefinition(self, node):

        # scope for locally-declared variables
        model_def_scope = ScopedMemory(
            enclosing_scope=self.model_scope
        )
        self.model_scope = model_def_scope

        # self.visit(node.arguments)
        self.visit(node.model)
        self.model_scope = self.model_scope.enclosing_scope


    def visit_Model(self, node):
        """
        Visiting each statement in the model definition. This doesn't check any
        of the optional arguments that can be invoked after the model command
        yet. Each statement is either a LocalModelDeclaration or ModelExpression
        """
        for single_statement in node._statements._list:
            self.visit(single_statement)


    def visit_LocalModelDeclaration(self, node):

        value = base.BaseEvaluator(
            tree=node.right,
            scope=self.model_scope
        )
        var_name = node.left.value
        self.model_scope.insert(var_name, value)

        self.outline._local_model_variables[var_name] = value


    def visit_ModelExpression(self, node):
        """
        joining the left and right model expressions into one expression, with
                        mexpr = left hand side - right hand side

        Then the joint mexpr is visited & records the period offsets of each of
        the endogenous variables in the model
        """
        mexpr = ast.BinaryOp(
            left=node.left,
            op=base.Token(base.MINUS, '-'),
            right=node.right
        )
        self.visit(mexpr)
        self.outline.add_model_expression_ast(mexpr)


    #
    # Model expressions record the periods of variables. SemanticAnalyzer has
    #    already checked that all variables were declared & only endogenous 
    #    variables have time offsets
    #
    def visit_BinaryOp(self, node):
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node):
        self.visit(node.expr)

    def visit_Num(self, node):
        # nothing to see here
        pass

    def visit_Var(self, node):
        var_name = node.value

        try:
            periods = self.outline._endo_lead_lags[var_name]
            periods.add(0)

        except KeyError:
            pass

    def visit_PeriodVar(self, node):
        var_name = node.value

        try:
            periods = self.outline._endo_lead_lags[var_name]
            periods.add(node.period_offset)

        except KeyError:
            pass

    def visit_Function(self, node):
        self.visit(node.expr)

    # 
    # Post-Model Block
    # 
    def visit_ModelActions(self, node):
        self.visit(node.boundaries)
        self.visit(node.shocks)


    def visit_ModelBoundaryValues(self, node):
        if node.initval:
            init_values = self.calculate_ModelBoundaryValues(node.initval)
            self.outline._initial_values = init_values

        if node.endval:
            end_values = self.calculate_ModelBoundaryValues(node.endval)
            self.outline._terminal_values = end_values

        if node.histval:
            hist_values = self.calculate_ModelBoundaryValues(node.histval)
            self.outline._historical_values = end_values


    def calculate_ModelBoundaryValues(self, boundaries):

        bound_dict = dict()

        # self.visit(boundaries.arguments)
        for assignment in boundaries._list:
            var_name = assignment.left.value
            var_value = base.BaseEvaluator(
                tree=assignment.right,
                scope=self.file_scope
            )

            bound_dict[var_name] = var_value

        return bound_dict


    def visit_ShocksBlock(self, node):

        shocks = {}

        # self.visit(node.arguments)
        for sdef in node._list:

            shock_type = sdef.shock_type
            moment = base.BaseEvaluator(
                tree=sdef.expr,
                scope=self.file_scope
            )

            sname = sdef.shock.value
            shock = shocks.get(sname, {})

            if shock_type == 'variance':
                shock['variance'] = moment

            elif shock_type == 'stderr':
                shock['variance'] = moment ** 2

            elif shock_type == 'correlated':
                oname = sdef.other.value
                shock['other'] = oname
                shock['corr'] = moment

            elif shock_type == 'covarying':
                oname = sdef.other.value
                shock['other'] = oname
                shock['cov'] = moment

            else:
                raise NotImplementedError(f"unrecognized shock type: {shock_type}")

            shocks[sname] = shock

        self.outline._shocks = shocks


    #
    # Utility
    #
    def visit_CompoundASTList(self, node):
        for single_ast in node._list:
            self.visit(single_ast)
