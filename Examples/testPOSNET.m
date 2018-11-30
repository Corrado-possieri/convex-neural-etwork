clear all
close all
clc

m = 100;
noi = 0; %1;

u = rand(1,m);

dgenfun = @(u) exp(exp(u/4).^2 + exp(u/3));

y = dgenfun(u);
y = max([y + noi*randn(size(y)); 1e-3*ones(size(y))]);

net = feedforwardnet(3,'trainlm');
tpam = net.trainParam;
tpam.mu_max = 1e2;
tpam.showWindow = false;

[net1, Temp1, netPar1, gpos] = trainPOS(u,y,12,'trainlm',tpam);

figure()
hold on
plot(u,y,'r*')
uu = linspace(min(u),max(u));
plot(uu,dgenfun(uu),'k--')
plot(uu,gpos(uu),'b')