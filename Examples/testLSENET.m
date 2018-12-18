clear all
close all
clc

m = 100;
noi = 0; %1e1;

u = 3*(rand(1,m)-1/2);

dgenfun = @(u) 1e-2*min(u,0).^6 + 8e-2*max(u,0).^4 + 10; %u.^4-2*u.^2+1;

y = dgenfun(u);
y = y + noi*randn(size(y));

net = feedforwardnet(3,'trainlm');
tpam = net.trainParam;
tpam.mu_max = 1e2;
tpam.showWindow = false;

[net1, Temp1, netPar1] = trainLSE(u,y,12,'trainlm',tpam);

figure()
hold on
plot(u,y,'r*')
uu = linspace(min(u),max(u));
plot(uu,dgenfun(uu),'k--')
plot(uu,net1(uu),'b')

Alpha = netPar1.Alpha;
Beta = netPar1.Beta;
uoff = netPar1.uoff;
umin = netPar1.umin;
Dugain = diag(netPar1.ugain);

cvx_begin
    variables x(1)
    minimize( log_sum_exp(Alpha*(Dugain*(x - uoff) + umin) + Beta))
cvx_end