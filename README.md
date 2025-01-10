# pynare

implementing the modeling and simulation features of the Dynare macroeconomic
modeling software in python.

this is mostly a project of passion, initially brought about by a sense of simultaneous
wonderment and despair when I first started to seriously work with Dynare -- how could
one set of code encompass all the macro models published since numerical models were first 
used to described the macroeconomy, and how could I ever hope to wrap my mind around it?
I have since shed that sense of naiveity and know Dynare can most certainly not handle
every type of model, although it is very capacious in its abilities. this repository is
my attempt at addressing the latter question: breaking Dynare, and modeling more
generally, down into small parts, coding it up, then moving onto the next small section.

I am aware of [A Comparison of Programming Languages in
Economics](https://www.nber.org/system/files/working_papers/w20263/w20263.pdf), the
[dolo](https://www.econforge.org/dolo.py/) project within the
[EconForge](https://www.econforge.org/) ecosystem, and the existing Matlab-wrapping
[pynare](https://github.com/gboehl/pynare) repository. until I eventually learn
cython and numba  I will not be able to rebut the first, the second has a head start
within an established programming network, and the third relies on Matlab/Dynare under the
hood and will thus always be up to date (in addition to having the first-mover advantage
when it comes to naming on `pip`). again I want to mention this is a project of passion
and intellectual satiation  rather than the rumblings of a revolution against high-cost
computing software (sorry, `Octave`) towards an open-source general-purpose programming
language, or anything so grandiose.


## capabilities

normally a `README` file has a `to-do` section which lists the few, or occasionally
several, things left to flesh out before the artist's vision is fully realized. I take
the complementary route here and instead list the few, and exactly few, things that
`pynare` can currently handle.
1. steady-state values. the steady state of a model can be found, provided its
dynamics are not sufficiently hairy.
2. linear solutions.
	- non-linear or global solutions are not addressed at all.
	- only first- and second-order solutions are available. k-th order solutions
are in the works
	- the standard Blanchard-Kahn checks are made
3. stochastic shocks:
	- only stochastic shocks are able to be simulated. invoking a deterministic
shock ought to throw a `NotImplementedError` somewhere.
4. simulation and impulse response plotting.
5. summaries of models, their solutions, and their statistical properties can be
automatically generated and printed to the command line


## example

in this example I plot the impulse response using the second-order solution to a
demand shock of a basic new-Keynesian model defined in
`pynare/examples/pynare/basic_nk.mod`. the minimum amount of code involving `pynare`
needed is just:
```python
import pynare as pn
import matplotlib.pyplot as plt

md = pn.Model.from_path('basic_nk')
imp = md.impulse_response(periods=30)

axes = imp.plot()
plt.show()
```
which generates the figure
![basic nk](https://github.com/loganhotz/pynare-dev/blob/master/readme_figs/basic_nk_impulse.png).

the abbreviated Dynare `.mod` file for this model is 
```
var n rr a y c w mu r pie;

...

model;
n = y + fixed_cost;  % Aggregate production function
w = chi * c * n^eta; % Labor Supply
w = 1/mu;            % Labor Demand

1 = r * beta * (a(+1)/a) * (c/c(+1)) * (1/pie(+1)); % Nominal Bond Euler Equation
1 = rr * beta * (a(+1)/a) * (c/c(+1));              % Real Bond Euler Equation
y = c + (phi_p/2) * (pie/piess - 1)^2 * y;          % National Accounts Identity

% Taylor-Rule
log(r) = log(rss) + phi_pie*log(pie/piess) + phi_y*log(y/yss);

% New-Keynesian Phillips Curve
phi_p*(pie/piess - 1)*(pie/piess) = (1-theta) + theta/mu + 
	beta * phi_p * (a(+1)/a) * (c/c(+1)) * (pie(+1)/piess-1) * (pie(+1)/piess) * (y(+1)/y);

% Preference Shock Evolution
a = rho_a*a(-1) + (1-rho_a)*ass + ea;
end;
```

## model descriptions
the model, along with its solution and statistical properties, can each be created in
a single line: `md.summary()`, `md.solution.summary()`, and `md.stats.summary()`. Using
the four-equation New-Keynesian model of Sims, Wu, and Zhang (2021) as an example,
calling these class methods would result in the following output to the command line:
```
model summary:

variables:
    number of variables:            8
    number of stochastic shocks:    4
    number of state vars:           4
    number of forward-looking vars: 4
    number of static vars:          2

steady-state eigenvalues:
       real    imag     mod
     0.5686  0.0000  0.5686
     0.8000  0.0000  0.8000
     0.8000  0.0000  0.8000
     0.8000  0.0000  0.8000
     1.1374  0.0000  1.1374
     1.2432  0.0000  1.2432
       -inf     nan     inf
        inf     nan     inf

covariance of stoch shocks:
                e_a e_theta    e_qe     e_r
    e_a      1.0000  0.0000  0.0000  0.0000
    e_theta  0.0000  1.0000  0.0000  0.0000
    e_qe     0.0000  0.0000  1.0000  0.0000
    e_r      0.0000  0.0000  0.0000  1.0000
```
and
```
statistics summary:

model moments:
              mean  std dev      var
    r       0.0000   0.0091   0.0001
    infl    0.0000   0.0122   0.0001
    y       0.0000   0.0262   0.0007
    yp      0.0000   0.0167   0.0003
    qe      0.0000   0.0167   0.0003
    theta   0.0000   0.0167   0.0003
    x       0.0000   0.0250   0.0006
    rn      0.0000   0.0050   0.0000

correlation:
                 r     infl        y       yp       qe    theta        x       rn
    r       1.0000  -0.8478  -0.9828  -0.2899   0.0115   0.0268  -0.8384   0.2899
    infl   -0.8478   1.0000   0.7835  -0.2589   0.0103   0.0240   0.9950   0.2589
    y      -0.9828   0.7835   1.0000   0.3902   0.0474   0.1106   0.7896  -0.3902
    yp     -0.2899  -0.2589   0.3902   1.0000   0.0000  -0.0000  -0.2569  -1.0000
    qe      0.0115   0.0103   0.0474   0.0000   1.0000   0.0000   0.0497   0.0000
    theta   0.0268   0.0240   0.1106  -0.0000   0.0000   1.0000   0.1161   0.0000
    x      -0.8384   0.9950   0.7896  -0.2569   0.0497   0.1161   1.0000   0.2569
    rn      0.2899   0.2589  -0.3902  -1.0000   0.0000   0.0000   0.2569   1.0000

autocovariance of endogenous variables:
             p = 0    p = 1    p = 2    p = 3    p = 4
    r       0.6056   0.3740   0.2363   0.1533   0.1024
    infl    0.5686   0.3233   0.1838   0.1045   0.0594
    y       0.6286   0.4054   0.2689   0.1836   0.1290
    yp      0.8000   0.6400   0.5120   0.4096   0.3277
    qe      0.8000   0.6400   0.5120   0.4096   0.3277
    theta   0.8000   0.6400   0.5120   0.4096   0.3277
    x       0.5715   0.3273   0.1880   0.1084   0.0628
    rn      0.8000   0.6400   0.5120   0.4096   0.3277

variance decomposition:
               e_a  e_theta     e_qe      e_r
    r       0.0985   0.0008   0.0002   0.9005
    infl    0.0817   0.0007   0.0001   0.9174
    y       0.1654   0.0123   0.0023   0.8200
    yp      1.0000   0.0000   0.0000   0.0000
    qe      0.0000   0.0000   1.0000   0.0000
    theta   0.0000   1.0000   0.0000   0.0000
    x       0.0805   0.0136   0.0025   0.9034
    rn      1.0000   0.0000   0.0000   0.0000
```
and finally,
```
first order policy and transition functions:
                     r     infl        y       yp       qe    theta        x       rn
    steady      0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000   0.0000
    r(-1)       0.5686  -0.7714  -1.5642   0.0000  -0.0000  -0.0000  -1.5642  -0.0000
    yp(-1)     -0.0691  -0.2303   0.3331   0.8000  -0.0000  -0.0000  -0.4669  -0.2388
    qe(-1)      0.0027   0.0091   0.0660   0.0000   0.8000   0.0000   0.0660   0.0000
    theta(-1)   0.0064   0.0213   0.1540   0.0000   0.0000   0.8000   0.1540   0.0000
                                                  +
    e_a        -0.0009  -0.0029   0.0042   0.0100  -0.0000  -0.0000  -0.0058  -0.0030
    e_theta     0.0001   0.0003   0.0019  -0.0000  -0.0000   0.0100   0.0019   0.0000
    e_qe        0.0000   0.0001   0.0008  -0.0000   0.0100  -0.0000   0.0008  -0.0000
    e_r         0.0071  -0.0096  -0.0196  -0.0000  -0.0000  -0.0000  -0.0196  -0.0000
```

the model and model statistics summaries each display 'blocks' of information about
its pertinent object; these can be toggled or edited by passing the relevant arguments
to each object's `summary()` method. see the code in `pynare.summary` for more detail.
