/*
 * This Dynare++ mod-file implements the RBC model with time-to-build
 * described in Kamenik (2011): "DSGE Models with Dynare++. A Tutorial."
 * Note that Dynare++ uses the same stock-at-the-end-of-period timing convention
 * as the regular Dynare
*/

var Y, C, K, A, H, B;

varexo EPS, NU;

parameters beta, rho, alpha, delta, theta, psi, tau;

alpha = 0.36;
rho = 0.95;
tau = 0.025;
beta = 1/(1.03^0.25);
delta = 0.025;
psi = 0;
theta = 2.95;

model;
  C*theta*H^(1+psi) = (1-alpha)*Y;
  beta*exp(B)*C/exp(B(1))/C(1)*
  (exp(B(1))*alpha*Y(1)/K(1)+1-delta) = 1;
  Y = exp(A)*K^alpha*H^(1-alpha);
  K = exp(B(-1))*(Y(-1)-C(-1)) + (1-delta)*K(-1);
  A = rho*A(-1) + tau*B(-1) + EPS;
  B = tau*A(-1) + rho*B(-1) + NU;
end;

initval;
  A = 0;
  B = 0;
  H = ((1-alpha)/(theta*(1-(delta*alpha)/(1/beta-1+delta))))^(1/(1+psi));
  Y = (alpha/(1/beta-1+delta))^(alpha/(1-alpha))*H;
  K = alpha/(1/beta-1+delta)*Y;
  C = Y - delta*K;
end;

vcov = [0.0002 0.00005;
0.00005 0.0001
];

order = 7;