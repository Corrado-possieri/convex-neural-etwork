clear all
close all
clc

m = 100;
noi = 0; %1e1;

u = 1e1*(rand(1,m)-1/2);

dgenfun = @(u) 1e-2*min(u,0).^6 + 8e-2*max(u,0).^4 + 10;

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