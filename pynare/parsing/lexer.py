"""
module for the pynare Lexer - reads in source code and generates a stream of
tokens reflecting the Dynare language. A pdf of the manual can be found here:
https://www.dynare.org/manual.pdf
"""



import pynare.parsing.base as base

from pynare.parsing.base import Token
from pynare.parsing.dynare import dynare_reserved_kw



class BaseLexer(object):
    """
    language-agnostic lexer
    """

    def __init__(
        self,
        text: str
    ):
        self.text = text

        self.line_pos = self.pos = 0
        self.current_line = 1 # I don't know any text editors that 0-index rows
        self.current_char = self.text[self.pos]


    def tokenize(self):
        self.token_pos = 0
        self.token_stream = list()

        while self.current_char is not None:
            t = self.next_token()
            self.token_stream.append(t)

        self.token_stream.append(Token(base.EOF, None, self.current_line, self.line_pos))


    def reset_lexer(self):
        self.token_pos = 0


    def error(self, symbol):
        if symbol:
            raise SyntaxError(
                f'unrecognized symbol: {repr(symbol)} in '
                f'line {self.current_line}, column {self.line_pos}'
            )


    def advance(self, n=1):
        self.pos += n
        self.line_pos += n
        try:
            if self.current_char == '\n':
                # moving line idx down to the next line, and resetting the line position 
                self.current_line += 1
                self.line_pos = 0
            self.current_char = self.text[self.pos]

        except IndexError:
            self.current_char = None


    def peek(self, n=1):
        peek_pos = self.pos + n
        try:
            return self.text[peek_pos]
        except IndexError:
            return None


    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()


    def _id(self):
        """
        recognizing characters or sequences of characters that may or may not be
        recognized keywords. If they are not, we tag them as a Token of type ID,
        and assume they're a variable
        """
        result = ''
        while self.valid_varchar(self.current_char):
            result += self.current_char
            self.advance()
        try:
            token = self.reserved_kw[result]
        except KeyError:
            token = base.reserved_funcs.get(result, Token(base.ID, result))
        token.assign_loc(self.current_line, self.line_pos)
        return token


    def _number(self):
        """
        recognizing numbers. We treat everything as a float, although Dynare (and
        Matlab underneath it) treats ints and floats as distinct types
        """
        result = ''
        while (
            self.current_char is not None
            and self.valid_numchar(self.current_char)
        ):
            result += self.current_char
            self.advance()
        num = float(result)
        return Token(base.NUMBER, num, self.current_line, self.line_pos)


    def _string(self, quote):
        """
        recognizing literal strings, which can be enclosed by single or double
        quotes in Dynare/Matlab.
        """
        result = ''
        self.advance()
        while self.current_char is not quote:
            result += self.current_char
            self.advance()
        self.advance()
        return Token(base.STRING, result, self.current_line, self.line_pos)


    def next_token(self):

        while self.current_char is not None:

            try:
                while self.current_char.isspace():
                    self.skip_whitespace()
            except AttributeError:
                if self.current_char is None:
                    return Token(base.EOF, None, self.current_line, self.line_pos)
                else:
                    self.error(self.current_char)

            while self.is_singleline_comment():
                self.skip_singleline_comment()

            while self.is_multiline_comment():
                self.skip_multiline_comment()

            if self.valid_begchar(self.current_char):
                return self._id()

            if self.valid_numchar(self.current_char):
                return self._number()


            #
            # Handling strings - not entirely sure what to do here. Matlab's 
            #    transpose character: " ' " can also be used make a character
            #    array. We tokenize any single quote as a string, so anyone
            #    who takes the transpose of a matrix is SOL for now
            #
            if self.current_char == '\'':
                return self._string(quote='\'')

            if self.current_char == '\"':
                return self._string(quote='\"')


            #
            # Grammar and Logical Symbols
            #
            if self.current_char == ',':
                self.advance()
                return Token(base.COMMA, ',', self.current_line, self.line_pos)

            if self.current_char == '.':
                self.advance()
                return Token(base.PERIOD, '.', self.current_line, self.line_pos)

            if self.current_char == ';':
                self.advance()
                return Token(base.SEMI, ';', self.current_line, self.line_pos)

            if self.current_char == ':':
                self.advance()
                return Token(base.COLON, ':', self.current_line, self.line_pos)

            if self.current_char == '#':
                self.advance()
                return Token(base.POUND, '#', self.current_line, self.line_pos)

            if self.current_char == '&':
                self.advance()
                return Token(base.AMPERSAND, '&', self.current_line, self.line_pos)

            if self.current_char == '|':
                self.advance()
                return Token(base.PIPE, '|', self.current_line, self.line_pos)

            if self.current_char == '@':
                self.advance()
                return Token(base.ATSIGN, '@', self.current_line, self.line_pos)

            if self.current_char == '!':
                self.advance()
                return Token(base.EXCLAMATION, '!', self.current_line, self.line_pos)

            if self.current_char == '~':
                self.advance()
                return Token(base.TILDE, '~', self.current_line, self.line_pos)

            if self.current_char == '%':
                self.advance()
                return Token(base.PERCENT, '%', self.current_line, self.line_pos)

            if self.current_char == '<':
                self.advance()
                return Token(base.LT, '<', self.current_line, self.line_pos)

            if self.current_char == '<' and self.peek() == '=':
                self.advance(2)
                return Token(base.LTOE, '<=', self.current_line, self.line_pos)

            if self.current_char == '>':
                self.advance()
                return Token(base.GT, '>', self.current_line, self.line_pos)

            if self.current_char == '>' and self.peek() == '=':
                self.advance(2)
                return Token(base.GTOE, '>=', self.current_line, self.line_pos)

            if self.current_char == '=' and self.peek() == '=':
                self.advance(2)
                return Token(base.EQUALITY, '==', self.current_line, self.line_pos)

            if self.current_char == '\\':
                self.advance()
                return Token(base.BACKSLASH, '\\', self.current_line, self.line_pos)


            # 
            # Common mathematical symbols
            # 
            if self.current_char == '+':
                self.advance()
                return Token(base.PLUS, '+', self.current_line, self.line_pos)

            if self.current_char == '-':
                self.advance()
                return Token(base.MINUS, '-', self.current_line, self.line_pos)

            if self.current_char == '*':
                self.advance()
                return Token(base.MUL, '*', self.current_line, self.line_pos)

            if self.current_char == '/':
                self.advance()
                return Token(base.DIV, '/', self.current_line, self.line_pos)

            if self.is_exponent():
                return self._exponent()

            if self.current_char == '(':
                self.advance()
                return Token(base.LPARE, '(', self.current_line, self.line_pos)

            if self.current_char == ')':
                self.advance()
                return Token(base.RPARE, ')', self.current_line, self.line_pos)

            if self.current_char == '[':
                self.advance()
                return Token(base.LBRACKET, '[', self.current_line, self.line_pos)

            if self.current_char == ']':
                self.advance()
                return Token(base.RBRACKET, ']', self.current_line, self.line_pos)

            if self.current_char == '{':
                self.advance()
                return Token(base.LBRACE, '{', self.current_line, self.line_pos)

            if self.current_char == '}':
                self.advance()
                return Token(base.RBRACE, '}', self.current_line, self.line_pos)

            if self.current_char == '=':
                self.advance()
                return Token(base.EQUALS, '=', self.current_line, self.line_pos)

            self.error(self.current_char)

        return Token(base.EOF, None, self.current_line, self.line_pos)


    def get_next_token(self):
        """
        Returns the token at the current token position (it's the next token from
        the Parser's perspective, however, hence the name), and iterates the token
        position by one
        """

        # if the token stream is ran through multiple times for some reason, we 
        #    need to reset the token position
        if self.token_pos >= len(self.token_stream):
            self.token_pos += 1 # self.token_pos = 0 ????

        token = self.token_stream[self.token_pos]
        self.token_pos += 1
        return token


    def see_future_token(self, k):
        future_pos = (self.token_pos - 1) + k
        if future_pos > len(self.token_stream) - 1:
            raise SyntaxError('Trying to look too far into the future.')
        else:
            return self.token_stream[future_pos]


    def see_next_token(self):
        """
        Returns the token at the current token position (it's the next token from
        the Parser's perspective, however, hence the name), but unlike
        self.get_next_token, the position is not incremented
        """
        return self.see_future_token(k=0)



class DynareLexer(BaseLexer):

    reserved_kw = dynare_reserved_kw

    def __init__(
        self,
        text: str
    ):

        super().__init__(text)

    def valid_varchar(self, c):
        # dynare variable names can have numbers, letters, or underscores
        try:
            return c.isalnum() or c == '_'
        except AttributeError:
            return False

    def valid_begchar(self, c):
        # dynare/matlab varnames can only begin with letters
        try:
            return c.isalpha()
        except AttributeError:
            return False

    def valid_numchar(self, c):
        # numbers can either begin with a decimal or number 
        try:
            return c.isdigit() or c == '.'
        except AttributeError:
            return False

    def is_singleline_comment(self):
        # single line comments are signified by '%' or '//'
        if (
            (self.current_char == '%')
            or (self.current_char == '/' and self.peek() == '/')
        ):
            return True
        return False


    def skip_singleline_comment(self):
        """ single line comments are signified by '%' or '//' """
        while self.current_char != '\n':
            self.advance()
        self.skip_whitespace() # just in case


    def is_multiline_comment(self):
        if self.current_char == '/' and self.peek() == '*':
            return True
        return False

    def skip_multiline_comment(self):
        # multiline comments begin '/*' and end '*/' 
        self.advance(n=2)
        while (self.current_char != '*') and (self.peek() != '/'):
            self.advance()

        # skipping the closing '*/'
        self.advance(n=2)
        self.skip_whitespace() # just in case

    def is_exponent(self):
        if self.current_char == '^':
            return True
        return False

    def _exponent(self):
        # is_exponent makes decision based on matlab's exponentiation symbol,
        #    but we use python's exponentiation symbol for the token
        self.advance()
        return Token(base.POWER, '**', self.current_line, self.line_pos)



class ExprLexer(object):
    """
    lexer that only handles simple mathematical expressions. recognizes numbers,
    variable names, the standard set of functions, and comparative operators.

    - the symbols '**' and '^' are both treated as exponents
    - variable names follow python's pattern; ie they start with underscores or
        letters
    """

    def __init__(
        self,
        text: str
    ):
        self.text = text

        self.line_pos = self.pos = 0
        self.current_line = 1 # I don't know any text editors that 0-index rows
        self.current_char = self.text[self.pos]


    def tokenize(self):
        self.token_pos = 0
        self.token_stream = list()

        while self.current_char is not None:
            t = self.next_token()
            self.token_stream.append(t)

        self.token_stream.append(Token(base.EOF, None, self.current_line, self.line_pos))


    def reset_lexer(self):
        self.token_pos = 0


    def error(self, symbol):
        if symbol:
            raise SyntaxError(
                f'unrecognized symbol: {repr(symbol)} in '
                f'line {self.current_line}, column {self.line_pos}'
            )


    def advance(self, n=1):
        self.pos += n
        self.line_pos += n
        try:
            if self.current_char == '\n':
                # moving line idx down to the next line, and resetting the line position 
                self.current_line += 1
                self.line_pos = 0
            self.current_char = self.text[self.pos]

        except IndexError:
            self.current_char = None


    def peek(self, n=1):
        peek_pos = self.pos + n
        try:
            return self.text[peek_pos]
        except IndexError:
            return None


    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()


    def valid_varchar(self, c):
        if c.isalpha() or c == '_':
            return True
        else:
            return False


    def valid_numchar(self, c):
        if c.isdigit() or c == '.':
            return True
        else:
            return False


    def _id(self):
        """
        recognizing characters or sequences of characters that may or may not be
        recognized keywords. If they are not, we tag them as a Token of type ID,
        and assume they're a variable
        """
        result = ''
        while self.valid_varchar(self.current_char):
            result += self.current_char
            self.advance()
        try:
            token = self.reserved_kw[result]
        except KeyError:
            token = base.reserved_funcs.get(result, Token(base.ID, result))
        token.assign_loc(self.current_line, self.line_pos)
        return token


    def _number(self):
        """
        recognizing numbers. We treat everything as a float, although Dynare (and
        Matlab underneath it) treats ints and floats as distinct types
        """
        result = ''
        while (
            self.current_char is not None
            and self.valid_numchar(self.current_char)
        ):
            result += self.current_char
            self.advance()
        num = float(result)
        return Token(base.NUMBER, num, self.current_line, self.line_pos)


    def next_token(self):

        while self.current_char is not None:

            try:
                while self.current_char.isspace():
                    self.skip_whitespace()
            except AttributeError:
                if self.current_char is None:
                    return Token(base.EOF, None, self.current_line, self.line_pos)
                else:
                    self.error(self.current_char)

            if self.valid_varchar(self.current_char):
                return self._id()

            if self.valid_numchar(self.current_char):
                return self._number()

            #
            # operators
            #
            if self.current_char == '<':
                self.advance()
                return Token(base.LT, '<', self.current_line, self.line_pos)

            if self.current_char == '<' and self.peek() == '=':
                self.advance(2)
                return Token(base.LTOE, '<=', self.current_line, self.line_pos)

            if self.current_char == '>':
                self.advance()
                return Token(base.GT, '>', self.current_line, self.line_pos)

            if self.current_char == '>' and self.peek() == '=':
                self.advance(2)
                return Token(base.GTOE, '>=', self.current_line, self.line_pos)

            if self.current_char == '=' and self.peek() == '=':
                self.advance(2)
                return Token(base.EQUALITY, '==', self.current_line, self.line_pos)

            # 
            # common mathematical symbols
            # 
            if self.current_char == '*' and self.peak() == '*':
                self.advance(2)
                return Token(base.POWER, '**', self.current_line, self.line_pos)

            if self.current_char == '^':
                self.advance()
                return Token(base.POWER, '**', self.current_line, self.line_pos)

            if self.current_char == '+':
                self.advance()
                return Token(base.PLUS, '+', self.current_line, self.line_pos)

            if self.current_char == '-':
                self.advance()
                return Token(base.MINUS, '-', self.current_line, self.line_pos)

            if self.current_char == '*':
                self.advance()
                return Token(base.MUL, '*', self.current_line, self.line_pos)

            if self.current_char == '/':
                self.advance()
                return Token(base.DIV, '/', self.current_line, self.line_pos)

            if self.current_char == '(':
                self.advance()
                return Token(base.LPARE, '(', self.current_line, self.line_pos)

            if self.current_char == ')':
                self.advance()
                return Token(base.RPARE, ')', self.current_line, self.line_pos)

            if self.current_char == '=':
                self.advance()
                return Token(base.EQUALS, '=', self.current_line, self.line_pos)

            self.error(self.current_char)

        return Token(base.EOF, None, self.current_line, self.line_pos)


    def get_next_token(self):
        """
        Returns the token at the current token position (it's the next token from
        the Parser's perspective, however, hence the name), and iterates the token
        position by one
        """

        # if the token stream is ran through multiple times for some reason, we 
        #    need to reset the token position
        if self.token_pos >= len(self.token_stream):
            self.token_pos += 1 # self.token_pos = 0 ????

        token = self.token_stream[self.token_pos]
        self.token_pos += 1
        return token


    def see_future_token(self, k):
        future_pos = (self.token_pos - 1) + k
        if future_pos > len(self.token_stream) - 1:
            raise SyntaxError('Trying to look too far into the future.')
        else:
            return self.token_stream[future_pos]


    def see_next_token(self):
        """
        Returns the token at the current token position (it's the next token from
        the Parser's perspective, however, hence the name), but unlike
        self.get_next_token, the position is not incremented
        """
        return self.see_future_token(k=0)
