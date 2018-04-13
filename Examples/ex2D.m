clear all
close all
clc

n = 2;
m = 1e2;

u = 1e1*(rand(n,m)-1/2);
y = u(1,:).^2 + u(2,:).^2;
y = y + 1e0*randn(size(y));

net = convexnet(u,y);

[net2,a,e,pf] = train(net,u,y);

xx = linspace(min(u(1,:)),max(u(1,:)),2e1);
yy = linspace(min(u(2,:)),max(u(2,:)),2e1);
[xxx,yyy] = meshgrid(xx,yy);
zzz = zeros(size(xxx));
for i1 = 1:size(xxx,1)
    for i2 = 1:size(xxx,2)
        px = [xxx(i1,i2); yyy(i1,i2)];
        zzz(i1,i2) = net2(px);
    end
end

hold on
surf(xxx,yyy,zzz)
plot3(u(1,:),u(2,:),y,'r*')
view([12,12])