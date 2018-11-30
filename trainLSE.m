function varargout = trainLSE(u,y,hiddenSize,trainFcn,trainPar)
%TRAINLSE train an LSE neural network
%
% [net, Temp, netPar] = TRAINLSE(u,y,hiddenSize,trainFcn,trainPar) takes 
% u: MxN input vector,
% y: N output vector, 
% hiddenSize: number of nodes in the hidden layer,
% trainFcn: training function, 
% trainPar: training parameters, 
% and returns 
% net: a trained LSET neural network  
% Temp: the temperature parameter of the network
% netPar: the parameters of the LSET network
%
% Defaults are used if TRAINLSE is called with fewer argument:
% hiddenSize = 10
% trainFcn = 'trainlm'

switch nargin
    case 0
        error 'the function requires the input and output vectors'
    case 1
        error 'the function requires the input and output vectors'
    case 2
        net = convexnet(u, y);
    case 3
        net = convexnet(u, y, hiddenSize);
    case 4
        net = convexnet(u, y, hiddenSize, trainFcn);
    case 5
        net = convexnet(u, y, hiddenSize, trainFcn);
        net.trainParam = trainPar;
end

[net,~,~,~] = train(net, u, y);

switch nargout
    case 1 
        varargout{1} = net;
    case 2
        varargout{1} = net;
        Temp = 1/net.outputs{2}.processSettings{1}.gain;
        varargout{2} = Temp;
    case 3
        varargout{1} = net;
        Temp = 1/net.outputs{2}.processSettings{1}.gain;
        varargout{2} = Temp;
        Alpha = net.IW{1,1};
        Beta = net.b{1};

        uoff = net.inputs{1}.processSettings{1}.xoffset;
        ugain = net.inputs{1}.processSettings{1}.gain;
        umin = net.inputs{1}.processSettings{1}.ymin;

        yoff = net.outputs{2}.processSettings{1}.xoffset;
        ygain = net.outputs{2}.processSettings{1}.gain;
        ymin = net.outputs{2}.processSettings{1}.ymin;

        
        netPar = struct('Alpha',Alpha,'Beta',Beta,'uoff',uoff,...
            'ugain',ugain,'umin',umin,'yoff',yoff,'ygain',...
            ygain,'ymin',ymin);
        varargout{3} = netPar;
end

end