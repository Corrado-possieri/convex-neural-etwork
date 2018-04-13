clear all
close all
clc

n = 1;
m = 1e2;

u = 1e1*(rand(n,m)-1/2);
y = 1e-2*min(u,0).^6 -u.^2 + 8e-2*max(u,0).^4;
y = y + randn(size(y));

net = convexnet(u,y,3);

[net2,a,e,pf] = train(net,u,y);

xx = linspace(min(u(1,:)),max(u(1,:)),2e2);
yy = net2(xx);

figure()
hold on
plot(xx,yy)
plot(u,y,'r*')