var c, k;

varexo x; % y;
/* Testing
a multiline
comment
*/
parameters aa alph bet delt gam;

aa = 0.8;
alph = 0.33;
bet = 0.9983;
gam = 2;
delt = 0.0005;
 
% single comment 

model;

[name = 'consumption', mcp = "r > -1.94478", id = 'CCC']
c = -k + aa*x*k(-1)^alph + (1-delt)*k(-1);

[dynamic]
c^(-gam) = (aa*alph*x(+1)*k^(alph-1) + 1 - delt) * c(+1)^(-gam)/(1+bet);
end;


// setting the initial values for endogenous variables
initval(all_values_required);
c = 0.7;
k = 0.4;
end;



shocks;
var x = 4;
% var y; stderr 1 + log(2);
end;
