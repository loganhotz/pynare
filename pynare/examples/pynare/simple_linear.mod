/*
Simple linear model found on page 23 of Dynare manual
*/

var x y;
varexo ex, ey;
parameters a b, d;

a = 0.25;
b = 2*a;
d = a/2;

model(linear);
x = a*x(-1) + b*y(+1) + ex;
y = d * y(-1) + ey;
end;
