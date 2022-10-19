"""
pynare errors
"""
from __future__ import annotations



class PynareError(Exception):
	pass



class PynareWarning(object):
	def __init__(self):
		pass

	def issue_warning(self, msg):
		issued = ': '.join([type(self).__name__, msg])
		print(issued)



class PynareSyntaxError(PynareError):
	def __init__(self, token):
		msg = (
			f"syntax error: {repr(token.value)} in line "
			f"{token.line_number} in column {token.line_pos}"
		)
		super().__init__(msg)



class ModelIdentificationError(PynareError):
	def __init__(self, bk_type=None, n0=None, n1=None):

		if bk_type == 'order':
			msg = (
				"model does not satisfy Blanchard-Kahn (1980) order condition: "
				f"{n0} explosive eigenvalues and {n1} forward-facing variables"
			)

		elif bk_type == 'rank':
			msg = "model does not satisfy Blanchard-Kahn (1980) rank condition"

		else:
			msg = (
				f"rank condition not met: model has rank {n0} "
				f"and there are {n1} static variables"
			)

		super().__init__(msg)



class EndogIndexError(PynareError, IndexError):

    def __init__(self, msg):
        super().__init__(msg)



class ModelVarError(PynareError):

    def __init__(self, msg):
        super().__init__(msg)



class CalibrationError(PynareError):

    def __init__(self, n: int):
        p = 'parameters' if n > 1 else 'parameter' # pedant
        msg = f"could not solve parameter calibration: {n} unsolved {p}"
        super().__init__(msg)
