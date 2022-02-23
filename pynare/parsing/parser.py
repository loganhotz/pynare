"""
language parsers
"""

from __future__ import annotations

from typing import List, Union, Tuple

import pynare.parsing.ast as ast
import pynare.parsing.base as base
import pynare.parsing.dynare as dyn

from pynare.parsing.base import Token
from pynare.errors import PynareSyntaxError



class BaseParser(object):
    """
    parser class for basic parser functions - type checking, peeking at
    future tokens, raising errors. Also has mathematical expression
    parsing functionality that would be shared across all languages
    """

    def __init__(
        self,
        lexer: BaseLexer
    ):
        self.lexer = lexer

        try:
            self.current_token = self.lexer.get_next_token()
        except AttributeError:
            self.lexer.tokenize()
            self.current_token = self.lexer.get_next_token()

    def error(self):
        raise PynareSyntaxError(self.current_token)

    def eat(self, token_type):
        """
        Compare the current token type with the passed token type and if they
        match, assign the next token to current token. Otherwise, error
        """
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def peek(self, k=0):
        return self.lexer.see_future_token(k=k)

    def peek_type(self, k=0):
        return self.peek(k=k).type

    def atom(self) -> AST:
        """
        atom : ID
             | NUMBER
             | LPARE expr RPARE
             | (NIMUS | PLUS) term
        """
        token = self.current_token

        if token.type == base.ID:
            self.eat(base.ID)
            return ast.Var(token)

        elif token.type == base.NUMBER:
            self.eat(base.NUMBER)
            return ast.Num(token)

        elif token.type == base.LPARE:
            self.eat(base.LPARE)
            node = self.expr()
            self.eat(base.RPARE)
            return node

        elif token.type in (base.MINUS, base.PLUS):
            self.eat(token.type)
            # sending 'expr' to term() instead of expr() is what ensures
            #    the proper unary operation functionality
            return ast.UnaryOp(
                op=token,
                expr=self.term()
            )

        self.error()

    def call(self) -> AST:
        """
        call : atom
             | FUNCTION LPARE expr RPARE

        we don't call the lower-level method first like we do in other expression
        methods, because the atom() method doesn't recognize tokens of type
        FUNCTION. Instead we do the try ... except below
        """
        while self.current_token.type == base.FUNCTION:
            token = self.current_token
            self.eat(base.FUNCTION)
            self.eat(base.LPARE)

            node = ast.Function(
                token=token,
                expr=self.expr()
            )

            self.eat(base.RPARE)

        try:
            return node
        except UnboundLocalError:
            return self.atom()

    def exponent(self) -> AST:
        """
        exponent : call (POWER exponent)*
        """
        node = self.call()

        while self.current_token.type == base.POWER:
            token = self.current_token
            self.eat(base.POWER)

            # sending the 'right' attribute to exponent() instead of atom()
            #    like other binary operations is what ensures exponentiation 
            #    to be right-associative
            return ast.BinaryOp(
                left=node,
                op=token,
                right=self.exponent()
            )

        return node

    def term(self) -> AST:
        """
        term : exponent ((MUL | DIV) exponent)*
        """
        node = self.exponent()

        while self.current_token.type in (base.MUL, base.DIV):
            token = self.current_token
            self.eat(token.type)

            node = ast.BinaryOp(
                left=node,
                op=token,
                right=self.exponent()
            )

        return node

    def expr(self) -> AST:
        """
        expr : term ((PLUS | MINUS) term)*
        """
        node = self.term()

        while self.current_token.type in (base.PLUS, base.MINUS):
            token = self.current_token
            self.eat(token.type)

            node = ast.BinaryOp(
                left=node,
                op=token,
                right=self.term()
            )

        return node

    def parse(self) -> AST:
        return self.expr()



class DynareParser(BaseParser):

    def __init__(
        self,
        lexer: BaseLexer
    ):
        super().__init__(lexer)

    def parse(self) -> ModelFile:
        return self.model_file()

    def model_file(self) -> ModelFile:

        model_variables = self.variable_preamble()
        model_definition = self.model_definition()
        model_actions = self.general_model_actions()

        return ast.ModelFile(
            model_preamble=model_variables,
            model_definition=model_definition,
            model_actions=model_actions
        )



    #
    # Declaring variables and assigning parameter values in the beginning 
    #    of the mod file
    #
    def variable_preamble(self) -> VariablePreamble:
        """
        The variable preamble is the entirety of mod file that comes before the
        model definition. Two main things are done in that section:
            1) Declaring variable and their types
            2) Assigning values to the parameters
        The parser doesn't do any type checks - it doesn't ensure variables
        being assigned values are actually parameters, nor does it ensure that
        any endogenous variables are declared


        variable_preamble : variable_type_declaration (variable_preamble)*
                          | parameter assignment (variable_preamble)*
                          | empty
        """

        # the change_type command that's listed in the same section of the dynare
        #    module is not included here - I really don't know how that's used yet
        dynare_variable_types = [
            dyn.VAR, dyn.VAREXO, dyn.VAREXO_DET, dyn.PARAMETERS,
            dyn.PREDETERMINED, dyn.TREND_VAR, dyn.LOG_TREND_VAR
        ]

        declaration_list = ast.CompoundASTList()
        parameter_list = ast.CompoundASTList()
        while self.current_token.type is not dyn.MODEL:

            if self.current_token.type in dynare_variable_types:
                declared_variables = self.variable_type_declaration()
                declaration_list.extend(declared_variables)

            elif self.current_token.type == base.ID:
                assigned_parameter = self.variable_assignment()
                parameter_list.append(assigned_parameter)

        return ast.VariablePreamble(
            declared_variables=declaration_list,
            assigned_parameters=parameter_list
        )

    def variable_type_declaration(self) -> List[VarDeclaration]:
        """
        variable_type_declaration : VARTYPE ID (COMMA ID)* SEMI
                                  : VARTYPE ID (ID)* SEMI

        declaring a particular type of variables, e.g.
            var y k c;
            var y, n, c, i;
        """
        declared_type = self.current_token
        self.eat(declared_type.type)

        variables_of_type = list()
        while self.current_token.type in (base.COMMA, base.ID):

            if self.current_token.type == base.COMMA:
                # commas are optional in variable declaration
                self.eat(base.COMMA)

            variables_of_type.append(self.variable())

        self.eat(base.SEMI)

        # each of the elements in variables_of_type are simple Var ASTs - we
        #    want to remember their declared types
        return [ast.VarDeclaration(v, declared_type) for v in variables_of_type]

    def variable_assignment(self) -> VarAssignment:
        """
        variable_assignment : ID EQUALS expr SEMI
        """
        var_node = self.variable()

        equals_token = self.current_token
        self.eat(base.EQUALS)

        assigned_value = self.expr()
        self.eat(base.SEMI)

        return ast.VarAssignment(
            left=var_node,
            op=equals_token,
            right=assigned_value
        )

    #
    # regular and period variables
    #
    def variable(self) -> Var:
        """
        Quick function for translating variable token to Var AST
        """
        node = ast.Var(self.current_token)
        self.eat(base.ID)
        return node

    def maybe_period_variable(self) -> Union[Var, PeriodVar]:
        """
        maybe_period_variable : ID
                              | ID period_offset
        returns ast.PeriodVar if offset given, ast.Var node otherwise
        """
        var = self.variable()
        if self.peek_type() == base.LPARE:
            direction, offset = self.period_offset()
            return ast.PeriodVar(
                var=var,
                direction=direction,
                offset=offset
            )

        return var

    def period_offset(self) -> Tuple[Token, Token]:
        """
        period_offset : LPARE (PLUS | MINUS) NUMBER RPARE
        """
        self.eat(base.LPARE)
        if self.current_token.type in (base.PLUS, base.MINUS):
            direction = self.current_token
            self.eat(direction.type)
        else:
            self.error()
        periods = self.current_token
        self.eat(base.NUMBER)
        self.eat(base.RPARE)

        return direction, periods

    #
    # Defining the Model
    #
    def model_definition(self) -> ModelDefinition:
        """
        model_definition : MODEL SEMI model END SEMI
                         | MODEL LPARE argument_list RPARE SEMI model END SEMI
        """

        self.eat(dyn.MODEL)
        model_arguments = self.optional_argument_list()
        self.eat(base.SEMI)

        model = self.model()
        self.eat(dyn.END)
        self.eat(base.SEMI)

        return ast.ModelDefinition(
            model=model,
            arguments=model_arguments
        )

    def model(self) -> Model:
        """
        model : model_statement (model_statement)+
        """
        model = ast.Model()

        while self.current_token.type != dyn.END:
            statement = self.model_statement()
            model.append_statement(statement)

        return model

    def model_statement(self) -> Union[VarAssignment, ModelExpression]:
        """
        model_statement : local_declaration SEMI
                        | tag model_expression SEMI
                        | model_expression SEMI
        """
        if self.current_token.type == base.POUND:
            statement = self.local_declaration()

        elif self.current_token.type == base.LBRACKET:
            tag = self.tag_list()
            statement = self.model_expression()
            statement.add_tag(tag)

        else:
            statement = self.model_expression()

        self.eat(base.SEMI)
        return statement

    def model_expression(self) -> ModelExpression:
        """
        model_expression : mexpr
                         | mexpr EQUALS mexpr
        """
        left = self.mexpr()
        if self.peek_type() == base.EQUALS:
            self.eat(base.EQUALS)
            right = self.mexpr()

        else:
            # if there's no equals sign, it's assumed this is a homogenous equation
            zero = Token(base.NUMBER, 0)
            right = ast.Num(zero)

        return ast.ModelExpression(
            left=left,
            right=right
        )

    def tag_list(self) -> CompoundASTList:
        """
        tag_list : LBRACKET single_tag (COMMA single_tag)* RBRACKET
        """
        self.eat(base.LBRACKET)
        tags = ast.CompoundASTList()
        tags.append(self.single_tag())

        while self.current_token.type == base.COMMA:
            self.eat(base.COMMA)
            tags.append(self.single_tag())
        self.eat(base.RBRACKET)

        return tags

    def single_tag(self) -> Argument:
        """
        single_tag : ID EQUALS STRING
                   | ID     <- only if its one of the steady tags
        """
        key = self.current_token
        self.eat(base.ID)

        if self.current_token.type == base.EQUALS:
            self.eat(base.EQUALS)
            value = self.current_token
            self.eat(base.STRING)
            return ast.Argument(
                token=key,
                value=value
            )

        return ast.Argument(token=key)

    def local_declaration(self) -> VarAssignment:
        """
        local_declaration : POUND ID EQUALS mexpr
        """
        self.eat(base.POUND)

        var_node = ast.Var(self.current_token)
        self.eat(base.ID)

        token = self.current_token
        self.eat(base.EQUALS)

        right = self.mexpr()

        return ast.LocalModelDeclaration(
            left=var_node,
            op=token,
            right=right
        )

    # 
    # the mathematical expressions in the model. these are identical to the 
    #    similarly named functions in the BaseParser, with the only difference
    #    being the 'matom' function can parse PeriodVars and Vars
    #
    def matom(self) -> AST:
        token = self.current_token

        if token.type == base.ID:
            var = self.maybe_period_variable()
            return var

        elif token.type == base.NUMBER:
            self.eat(base.NUMBER)
            return ast.Num(token)

        elif token.type == base.LPARE:
            self.eat(base.LPARE)
            node = self.mexpr()
            self.eat(base.RPARE)
            return node

        elif token.type in (base.MINUS, base.PLUS):
            self.eat(token.type)
            node = ast.UnaryOp(
                op=token,
                expr=self.mterm()
            )
            return node

        self.error()

    def mcall(self) -> AST:
        while self.current_token.type == base.FUNCTION:
            token = self.current_token
            self.eat(base.FUNCTION)
            self.eat(base.LPARE)
            node = ast.Function(
                token=token,
                expr=self.mexpr()
            )
            self.eat(base.RPARE)

        try:
            return node
        except UnboundLocalError:
            return self.matom()

    def mexponent(self) -> AST:
        node = self.mcall()

        while self.current_token.type == base.POWER:
            token = self.current_token
            self.eat(base.POWER)

            node = ast.BinaryOp(
                left=node,
                op=token,
                right=self.mexponent()
            )

        return node

    def mterm(self) -> AST:
        node = self.mexponent()

        while self.current_token.type in (base.MUL, base.DIV):
            token = self.current_token
            self.eat(token.type)

            node = ast.BinaryOp(
                left=node,
                op=token,
                right=self.mexponent()
            )

        return node

    def mexpr(self) -> AST:
        node = self.mterm()

        while self.current_token.type in (base.PLUS, base.MINUS):
            token = self.current_token
            self.eat(token.type)

            node = ast.BinaryOp(
                left=node,
                op=token,
                right=self.mterm()
            )

        return node

    #
    # Commands for after model has been defined
    #
    def general_model_actions(self) -> ModelActions:
        """
        After the model definition has been parsed in model_definition, the
        simulation and stochastic commands follow. These commands include:
            1) Defining model boundary conditions in initval, endval, and histval
                blocks.
            2) Defining the shocks to the model in the shocks block
            3) General commands like 'steady;', 'homotopy_setup;', 'steady_state_model;'

        I don't anticipate trying to read matlab's plotting functions, so the end
        of the simulation_commands block is also the end of the .mod file
        """
        model_actions = ast.ModelActions()

        boundary_commands = [dyn.INITVAL, dyn.ENDVAL, dyn.HISTVAL]

        while self.current_token.type != base.EOF:

            if self.current_token.type in boundary_commands:
                bound_type = self.current_token.type
                bound_values = self.boundary_values()

                model_actions.set_boundary_values(bound_values, bound_type)

            elif self.current_token.type == dyn.SHOCKS:
                shocks = self.shocks_block()
                model_actions.set_shocks(shocks)

            else:
                generic_commands = self.model_actions()

        return model_actions

    def boundary_values(self) -> ArgCompoundASTList:
        """
        boundary_values : INITVAL SEMI (ID EQUALS expr SEMI)+ END SEMI
                        | INITVAL argument_list SEMI (ID EQUALS expr)+ END SEMI
                        | ENDVAL SEMI (ID EQUALS expr)+ END SEMI
                        | ENDVAL argument_list SEMI (ID EQUALS expr)+ END SEMI
                        | HISTVAL SEMI (ID EQUALS expr)+ END SEMI
                        | HISTVAL argument_list SEMI (ID EQUALS expr)+ END SEMI
        """

        # the 'if' statement in simulation_commands will have already ensured
        #    the current token is a valid boundary command
        self.eat(self.current_token.type)

        arguments = self.optional_argument_list()
        self.eat(base.SEMI)

        bound_values = ast.ArgCompoundASTList(arguments)

        while self.current_token.type != dyn.END:
            assigned_boundary = self.variable_assignment()
            bound_values.append(assigned_boundary)

        self.eat(dyn.END)
        self.eat(base.SEMI)
        return bound_values

    def shocks_block(self) -> ShocksBlock:
        self.eat(dyn.SHOCKS)

        arguments = self.optional_argument_list()
        self.eat(base.SEMI)

        shocks = ast.ShocksBlock(arguments)

        while self.current_token.type in (dyn.CORR, dyn.VAR):
            single_shock = self.shock()
            self.eat(base.SEMI)
            shocks.append(single_shock)

        self.eat(dyn.END)
        self.eat(base.SEMI)
        return shocks

    def shock(self) -> Union[StochasticShock, DeterministicShock]:
        """
        The dynare shock block has a large vocabulary used to specify how
        shocks to the model are administered, which makes parsing it a rather
        tedious job
        """

        if self.current_token.type == dyn.CORR:
            return self.correlated_shock()

        elif (
            self.current_token.type == dyn.VAR
            and self.peek_type(k=2) == base.COMMA
        ):
            return self.covarying_shock()

        elif (
            self.current_token.type == dyn.VAR
            and self.peek_type(k=2) == base.EQUALS
        ):
            return self.variance_shock()

        elif (
            self.current_token.type == dyn.VAR
            and self.peek_type(k=2) == base.SEMI
            and self.peek_type(k=3) == dyn.STDERR
        ):
            return self.stderr_shock()

        else:
            return self.deterministic_shock()

    def correlated_shock(self) -> StochasticShock:
        """
        correlated_shock : CORR ID COMMA ID EQUALS expr
        """
        self.eat(dyn.CORR)
        x = self.variable()

        self.eat(base.COMMA)
        y = self.variable()

        self.eat(base.EQUALS)
        expr = self.expr()

        return ast.StochasticShock(
            shock_type='correlated',
            shock=x,
            other=y,
            expr=expr
        )

    def covarying_shock(self) -> StochasticShock:
        """
        covarying_shock : VAR ID COMMA ID EQUALS expr
        """
        self.eat(dyn.VAR)
        x = self.variable()

        self.eat(base.COMMA)
        y = self.variable()

        self.eat(base.EQUALS)
        expr = self.expr()

        return ast.StochasticShock(
            shock_type='covarying',
            shock=x,
            other=y,
            expr=expr
        )

    def variance_shock(self) -> StochasticShock:
        """
        variance_shock : VAR ID EQUALS expr
        """
        self.eat(dyn.VAR)
        token = self.variable()

        self.eat(base.EQUALS)
        expr = self.expr()

        return ast.StochasticShock(
            shock_type='variance',
            shock=token,
            expr=expr
        )

    def stderr_shock(self) -> StochasticShock:
        """
        stderr_shock : VAR ID SEMI STDERR expr
        """
        self.eat(dyn.VAR)
        token = self.variable()

        self.eat(base.SEMI)
        self.eat(dyn.STDERR)

        expr = self.expr()

        return ast.StochasticShock(
            shock_type='stderr',
            shock=token,
            expr=expr
        )

    def deterministic_shock(self) -> DeterministicShock:
        """
        deterministic_shock : VAR ID SEMI PERIODS periods_list SEMI VALUES values_list
        """
        self.eat(dyn.VAR)
        token = self.variable()

        self.eat(base.SEMI)
        self.eat(dyn.PERIODS)
        periods = self.shock_periods_list()

        self.eat(base.SEMI)
        self.eat(dyn.VALUES)
        values = self.shock_values_list()

        return ast.DeterministicShock(
            shock_type='deterministic',
            shock=token,
            periods=periods,
            values=values
        )

    def shock_periods_list(self) -> CompoundASTList:
        """
        A single shock period can either be a single integer, signifying a one
        period deterministic shock, or a range of periods from a:b

        shock_periods_list : ((NUM | NUM:NUM) (COMMA))+
        """

        shock_periods = ast.CompoundASTList()

        while self.current_token.type in (base.COMMA, base.NUMBER):

            if self.current_token.type == base.COMMA:
                self.eat(base.COMMA)

            start = self.current_token
            self.eat(base.NUMBER)

            if self.current_token.type == base.COLON:
                self.eat(base.COLON)
                end = self.current_token
                self.eat(base.NUMBER)
                single_period = ast.DeterministicShockPeriod(
                    start=start,
                    end=end
                )

            else:
                single_period = ast.DeterministicShockPeriod(
                    start=start,
                    end=start
                )

            shock_periods.append(single_period)

        return shock_periods

    def shock_values_list(self) -> CompoundASTList:
        """
        The values of the shocks can be single numbers or general expressions.
        If the latter, the whole expression must be in parantheses

        shocK_values_list : ((NUM | LPARE expr RPARE) (COMMA))+

        """
        shock_values = ast.CompoundASTList()

        while self.current_token.type in (base.COMMA, base.NUMBER, base.LPARE):

            if self.current_token.type == base.COMMA:
                self.eat(base.COMMA)

            if self.current_token.type == base.NUMBER:
                token = self.current_token
                self.eat(base.NUMBER)
                single_value = ast.Num(token)

            else:
                self.eat(base.LPARE)
                single_value = self.expr()
                self.eat(base.RPARE)

            shock_values.append(single_value)

        return shock_values

    def model_actions(self):

        if self.current_token.type == base.ID:
            assigned_var = self.variable_assignment()

        elif self.current_token.type == dyn.STEADY:
            print('gonna find the steady state of the model')

        raise NotImplementedError(self.current_token)

    #
    # parsing argument lists
    #
    def optional_argument_list(self) -> CompoundASTList:
        """
        dynare allows for both keyword arguments and placement arguments, whereas
        matlab only uses the presence of arguments to signifiy deviations from
        the default

        optional_argument_list : LPARE (ID | ID EQUALS (ID | STRING | expr))+ RPARE
                               | empty
        """
        if self.current_token.type != base.LPARE:
            return ast.CompoundASTList()

        return self.argument_list()

    def argument_list(self) -> CompoundASTList:
        self.eat(base.LPARE)
        arg_list = ast.CompoundASTList()

        first_arg = self.single_argument()
        arg_list.append(first_arg)

        while self.current_token.type == base.COMMA:
            self.eat(base.COMMA)
            single_arg = self.single_argument()
            arg_list.append(single_arg)

        self.eat(base.RPARE)
        return arg_list

    def single_argument(self) -> Argument:
        """
        single_argument : ID
                        | ID EQUALS STRING
                        | ID EQUALS ID
                        | ID EQUALS expr
        """
        var_token = self.current_token
        self.eat(base.ID)

        if self.current_token.type == base.EQUALS:
            self.eat(base.EQUALS)

            if self.current_token.type == base.STRING:
                value = self.current_token
                self.eat(base.STRING)
                return ast.Argument(
                    token=var_token,
                    value=value
                )

            elif self.current_token.type == base.ID:
                value = self.current_token
                self.eat(base.ID)
                return ast.Argument(
                    token=var_token,
                    value=value
                )

            else:
                value = self.expr()
                return ast.Argument(
                    token=var_token,
                    value=value
                )

        return ast.Argument(
            token=var_token,
            value=None
        )
