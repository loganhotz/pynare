
from __future__ import annotations

from pynare.parsing.base import ABCVisitor

import pynare.parsing.dynare as dyn


class Symbol(object):

    # symbols hold then ame and variable type of the Var AST nodes encountered
    def __init__(self, name: str, stype: ModelTypeSymbol = None):
        self.name = name
        self.stype = stype


class ModelTypeSymbol(Symbol):

    # the language-specific builtin symbol types
    def __init__(self, name):
        super().__init__(name)

    def __repr__(self):
        return f'<ModelTypeSymbol({self.name})>'


class VarSymbol(Symbol):

    def __init__(self, name, stype):
        super().__init__(name, stype)

    def __repr__(self):
        return f'<VarSymbol({self.name}: {self.stype})>'


class ScopedSymbolTable(object):

    def __init__(
        self,
        scope_name: str,
        scope_level: int,
        enclosing_scope: ScopedSymbolTable = None
    ):
        self._symbols = dict()

        self.scope_name = scope_name
        self.scope_level = scope_level
        self.enclosing_scope = enclosing_scope
        self._init_builtins()

    def _init_builtins(self):
        pass

    def insert(self, symbol):
        self._symbols[symbol.name] = symbol

    def lookup(self, name):

        symbol = self._symbols.get(name)
        if symbol is not None:
            return symbol

        if self.enclosing_scope is not None:
            return self.enclosing_scope.lookup(name)

        return None

    def __str__(self):

        n_delims = 50
        h1 = 'SCOPE (Scoped Symbol Table)'
        lines = ['\n', h1, '='*n_delims]
        for header_name, header_value in (
            ('Scope Name', self.scope_name),
            ('Scope Level', self.scope_level),
            ('Enclosing Scope',
                self.enclosing_scope.scope_name if self.enclosing_scope else None
            )
        ):
            lines.append('{:<16}: {}'.format(header_name, header_value))

        h2 = 'Scope (Scoped Symbol Table) Contents'
        lines.extend([h2, '-'*n_delims])
        lines.extend(
            (('{:>16}: {}').format(k, v)
            for k, v in self._symbols.items()
            if not isinstance(v, ModelTypeSymbol))
        )
        lines.append('\n')
        return '\n'.join(lines)



"""
Dynare
"""
class DynareSymbolTable(ScopedSymbolTable):

    def __init__(
        self,
        scope_name: str,
        scope_level: int,
        enclosing_scope: ScopedSymbolTable = None
    ):
        super().__init__(scope_name, scope_level, enclosing_scope)

    def _init_builtins(self):
        """
        Dynare treats its variable scopes very oddly, in my opinion. You can
        declare variables & assign values before and after the model definition
        with abandon, but if they're used in the model definition they must be
        declared as parameters in the preamble.
            This seems a superfluous distinction to me; why not treat all symbols
        encountered in the model block as parameters by default, unless they are
        declared as endogenous/exogenous in the preamble? Regardless, we respect
        that distinction and assign those non-parameter, non-model variables the
        'non_model' type
        """
        self.insert(ModelTypeSymbol(dyn.VAR))
        self.insert(ModelTypeSymbol(dyn.VAREXO))
        self.insert(ModelTypeSymbol(dyn.VAREXO_DET))
        self.insert(ModelTypeSymbol(dyn.PARAMETERS))
        self.insert(ModelTypeSymbol(dyn.PREDETERMINED))
        self.insert(ModelTypeSymbol(dyn.TREND_VAR))
        self.insert(ModelTypeSymbol(dyn.LOG_TREND_VAR))
        self.insert(ModelTypeSymbol('non_model'))


class DynareSemanticAnalyzer(ABCVisitor):

    def __init__(
        self,
        parser: DynareParser
    ):
        super().__init__(parser)

        self.current_scope = DynareSymbolTable(
            scope_name='global',
            scope_level=1
        )


    def verify(self):
        self.visit(self.tree)


    def visit_ModelFile(self, node):
        self.visit(node.model_preamble)
        self.visit(node.model_definition)
        self.visit(node.model_actions)


    #
    # Pre-Model Block
    #
    def visit_VariablePreamble(self, node):
        self.visit(node.variables)
        self.visit(node.parameters)


    def visit_VarDeclaration(self, node):
        """
        retrieve the ModelSymbol based on 'vtype', create a VarSymbol based on
        that type and the name of the variable, and save to the global scope
        """

        # getting built-in type, e.g. 'var', 'varexo', 'parameters'
        vtype = node.vtype.value
        stype = self.current_scope.lookup(vtype)

        # variable name that's being declared & assigning its built-in type
        var_name = node.var_node.value
        var_symbol = VarSymbol(var_name, stype)

        # if this variable's name has already been used, throw an error
        if self.current_scope.lookup(var_name) is not None:
            token = node.var_node.token
            raise NameError(
                f'row {token.line_number}, column {token.line_pos}. '
                f'name {repr(var_name)} has already been declared'
            )

        self.current_scope.insert(var_symbol)


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
        var_name = node.left.value
        var_symbol = self.current_scope.lookup(var_name)

        if var_symbol is None:
            # assigning non-model parameters, validating the expression
            var_symbol = VarSymbol(var_name, 'non_model')
            self.current_scope.insert(var_symbol)
            self.visit(node.right)

        else:
            # we don't want to assign values to non-parameter values
            if var_symbol.stype.name != 'parameters':
                token = node.left.token
                raise TypeError(
                    f'row {token.line_number}, column {token.line_pos}. '
                    f'assigning value to non-parameter: {repr(var_name)}'
                )

            self.visit(node.right)


    #
    # Model Block
    #
    def visit_ModelDefinition(self, node):

        # scope for locally-declared variables
        model_scope = DynareSymbolTable(
            scope_name='model',
            scope_level=self.current_scope.scope_level+1,
            enclosing_scope=self.current_scope
        )
        self.current_scope = model_scope

        # self.visit(node.arguments)
        self.visit(node.model)

        self.current_scope = self.current_scope.enclosing_scope


    def visit_Model(self, node):
        for single_statement in node._statements._list:
            self.visit(single_statement)


    def visit_LocalModelDeclaration(self, node):
        # all local declarations are required to be parameters
        stype = self.current_scope.lookup(dyn.PARAMETERS)
        var_name = node.left.value
        var_symbol = VarSymbol(var_name, stype)

        # local declarations cannot collide with variable declarations made
        #    outside the model block
        if self.current_scope.lookup(var_name) is not None:
            token = node.left.token
            raise NameError(
                f'row {token.line_number}, column {token.line_pos}. '
                f'name {repr(var_name)} has already been declared outside model'
            )

        self.visit(node.right)
        self.current_scope.insert(var_symbol)


    def visit_ModelExpression(self, node):
        # no need to visit the tags (yet), just the left and right expressions
        self.visit(node.left)
        self.visit(node.right)


    def visit_CompoundASTList(self, node):
        for single_ast in node._list:
            self.visit(single_ast)



    # 
    # Post-Model Block
    #
    def visit_ModelActions(self, node):
        self.visit(node.boundaries)
        self.visit(node.shocks)


    def visit_ModelBoundaryValues(self, node):
        if node.initval:
            self.validate_ModelBoundaryValues(node.initval)

        if node.endval:
            self.validate_ModelBoundaryValues(node.endval)

        if node.histval:
            self.validate_ModelBoundaryValues(node.histval)


    def validate_ModelBoundaryValues(self, boundaries):

        # self.visit(boundaries.arguments)
        for assignment in boundaries._list:
            var_name = assignment.left.value
            var_symbol = self.current_scope.lookup(var_name)

            if var_symbol.stype.name != dyn.VAR:
                token = assignment.left.token
                raise TypeError(
                    f'row {token.line_number}, column {token.line_pos}. '
                    f'variable {repr(var_name)} was not declared as endogenous. '
                    'Cannot set boundary value'
                )

            # make sure expressions are valid
            self.visit(assignment.right)


    def _check_shock_declaration(self, shock_sym, shock_def):
        if shock_sym.stype.name not in (dyn.VAREXO, dyn.VAREXO_DET):
            token = shock_def.shock.token
            raise TypeError(
                f'row {token.line_number}, column {token.line_pos}. '
                f'variable {repr(shock_sym)} was not declared as exogenous. '
                'Cannot assign as a shock'
            )


    def visit_ShocksBlock(self, node):

        # self.visit(boundaries.arguments)
        for sdef in node._list:

            # check that shock was declared as exogenous
            shock_name = sdef.shock.value
            shock_symbol = self.current_scope.lookup(shock_name)
            self._check_shock_declaration(shock_symbol, sdef)

            # check that moment expression is valid
            self.visit(sdef.expr)

            # check correlated shock, if it exists
            if sdef.shock_type in ('correlated', 'covarying'):
                other_name = sdef.other.value
                other_symbol = self.current_scope.lookup(other_name)
                self._check_shock_declaration(other_symbol, sdef)


    #
    # Algebraic expression
    #
    def visit_BinaryOp(self, node):
        # checks for semantic errors in left and right exprs
        self.visit(node.left)
        self.visit(node.right)


    def visit_UnaryOp(self, node):
        # check for semantic errors in expr
        self.visit(node.expr)


    def visit_Num(self, node):
        # nothing to see here
        pass


    def visit_Var(self, node):
        # checks that the variable has been defined
        var_name = node.value
        var_symbol = self.current_scope.lookup(var_name)
        if var_symbol is None:
            token = node.token
            raise NameError(
                f'row {token.line_number}, column {token.line_pos}. '
                f'name {var_name} was not declared'
            )


    def visit_PeriodVar(self, node):
        var_name = node.value
        var_symbol = self.current_scope.lookup(var_name)

        if var_symbol is None:
            token = node.token
            raise NameError(
                f'row {token.line_number}, column {token.line_pos}. '
                f'name {var_name} was not declared'
            )

        if var_symbol.stype.name == dyn.PARAMETERS:
            token = node.token
            raise TypeError(
                f'row {token.line_number}, column {token.line_pos}. '
                f'parameters cannot be out-of-period'
            )


    def visit_Function(self, node):
        # check the argument of the function
        self.visit(node.expr)
