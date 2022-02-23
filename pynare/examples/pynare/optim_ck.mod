/*
Solving for the optimal path of consumption and capital. This model appears in 
the Dynare manual on page 28, and is accompanied by a nice description of how 
the initval and endval blocks work (which is why I've included it as an example)
*/


var c k;
varexo x;

parameters aa alph delt gam bet;
aa 		= 1.1;
alph 	= 0.66;
delt 	= 0.05;
gam 	= 0.04;
bet 	= 1/0.95 - 1;

model;
c + k - aa*x*k(-1)^alph - (1-delt)*k(-1);
c^(-gam) - (1+bet)^(-1)*(aa*alph*x(+1)*k^(alph-1) + 1 - delt)*c(+1)^(-gam);
end;

initval;
k = 12;
end;

endval;
c = 2;
x = 1.1;
end;

simul(periods=200);