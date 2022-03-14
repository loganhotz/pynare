
var n rr a y c w mu r pie;

varexo ea;

parameters chi eta phi_p phi_pie phi_y theta beta rho_a vol_a
	yss ass zss muss css wss fixed_cost nss
	piess rss rrss;


% Structural Parameters
beta 	= 0.99;
phi_p 	= 160;
eta 	= 1/1; % 1/frisch_elasticity
theta 	= 6;


% Steady-State Normalization & Parameter Implications
yss 		= 1;
ass 		= 1;
zss 		= 1;
muss 		= theta/(theta - 1);
css 		= yss;
wss 		= muss^(-1);
fixed_cost 	= (muss - 1)*yss;
nss 		= yss + fixed_cost;
chi 		= wss * css^(-1) * nss^(-eta);


% Monetary Policy Parameters
piess 	= 1;
rss 	= piess/beta;
rrss 	= 1/beta;
phi_pie = 2;
phi_y 	= 0.5;


% Stochastic Processes
rho_a = 0.90;
vol_a = 0.005;


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


initval;
a = ass;
n = nss;
pie = piess;
rr = rrss;
r = rss;
y = yss;
c = css;
w = wss;
mu = muss;
end;

shocks;
var ea = vol_a^2;
end;
