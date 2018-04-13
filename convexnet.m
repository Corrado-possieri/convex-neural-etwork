function outnet = convexnet(u,y,hiddenSize,trainFcn)
%CONVEXNET design of a convex artificial neural network
%
% CONVEXNET(u,y,hiddenSize,trainFcn) takes a MxN input vector,
% and N output vector, the number of nodes in the hidden layer,
% and a backpropagation training function, and returns
% a feed-forward convex neural network.
%
% Defaults are used if CONVEXNET is called with fewer argument:
% hiddenSize = 10
% trainFcn = 'trainlm'


switch nargin
    case 0
        error 'the function requires the input and output vectors'
    case 1
        error 'the function requires the input and output vectors'
    case 2
        net = feedforwardnet;
    case 3
        net = feedforwardnet(hiddenSize);
    case 4
        net = feedforwardnet(hiddenSize,trainFcn);
end

net1 = configure(net,'inputs',u);
net1 = configure(net1,'outputs',y);

net1.biasConnect = [1;0];

net1.LW{2,1} = ones(size(net1.LW{2,1}));
net1.layerWeights{2}.learn = 0;

net1.layers{1}.transferFcn = 'expfun';

net1.layers{2}.transferFcn = 'logfun';

outnet = net1;

end