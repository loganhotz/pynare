"""
forward-facing functions for parsing the text that defines models
"""

from __future__ import annotations

from pynare.parsing.base import BaseEvaluator
from pynare.parsing.parser import BaseParser
from pynare.parsing.factory import ModelFactory


def read_model(
    model_text: str,
    language: str
) -> AST:
    """
    The top-level function for translating a model from text -> AST. Called in
    the Model classmethod 'from_path'


    Parameters
    ----------
    model_text : str
        a list of the lines of the text that defines a model
    language : str
        the language the model is written in

    Returns
    -------
    Abstract Syntax Tree of the model's definition
    """
    return ModelFactory(model_text, language)



def parse_string(
    expr: str,
    exp: str = 'standard'
) -> AST:
    """
    a top-level function for translating simple mathematical expressions in string
    form to an AST

    Parameters
    ----------
    expr : str
        a mathematical expression
    exp : str ( = 'standard' )
        the paradigm for recognizing exponents in `expr`. accepted values are
        'standard', and 'python', which use '^' and '**' as their respective
        exponent signifiers

    Returns
    -------
    AST
    """
    parser = BaseParser(expr, exp=exp)
    return parser.parse()



def evaluate(
    expr: str | AST,
    scope: Mapping = {},
    exp: str = 'standard'
):
    """
    evaluate a simple mathematical expression involving algebraic operations, the
    reserved functions, and comparative operators

    Parameters
    ----------
    expr : str | AST
        a mathematical expression. can be a string or an already-parsed AST
    scope : Mapping
        a mapping of variable names in `expr` to their values
    exp : str ( = 'standard' )
        the paradigm for recognizing exponents in `expr`. accepted values are
        'standard', and 'python', which use '^' and '**' as their respective
        exponent signifiers

    Returns
    -------
    int | float | bool
    """
    if isinstance(expr, str):
        ast = parse_string(expr, exp=exp)
        return BaseEvaluator(ast, scope=scope)
    return BaseEvaluator(expr, scope=scope)
