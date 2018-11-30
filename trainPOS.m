function varargout = trainPOS(u,y,hiddenSize,trainFcn,trainPar)
%TRAINPOS train a posynomial neural network
%
% [net, Temp, netPar, gpos] = TRAINPOS(u,y,hiddenSize,trainFcn,trainPar) takes 
% u: MxN input vector,
% y: N output vector, 
% hiddenSize: number of nodes in the hidden layer,
% trainFcn: training function, 
% trainPar: training parameters,
% and returns 
% net: a trained GPOS neural network  
% Temp: the temperature parameter of the network
% netPar: the parameters of the network
% gpos: the generalized posynomial
%
% Defaults are used if TRAINPOS is called with fewer argument:
% hiddenSize = 10
% trainFcn = 'trainlm'

switch nargin
    case 0
        error 'the function requires the input and output vectors'
    case 1
        error 'the function requires the input and output vectors'
    case 2
        if ~isempty(find(u <= 0, 1))
            error 'the inputs must be positive'
        end
        if ~isempty(find(y <= 0, 1))
            error 'the outputs must be positive'
        end
        ut = log(u);
        yt = log(y);
        net = convexnet(ut, yt);
    case 3
        if ~isempty(find(u <= 0, 1))
            error 'the inputs must be positive'
        end
        if ~isempty(find(y <= 0, 1))
            error 'the outputs must be positive'
        end
        ut = log(u);
        yt = log(y);
        net = convexnet(ut, yt, hiddenSize);
    case 4
        if ~isempty(find(u <= 0, 1))
            error 'the inputs must be positive'
        end
        if ~isempty(find(y <= 0, 1))
            error 'the outputs must be positive'
        end
        ut = log(u);
        yt = log(y);
        net = convexnet(ut, yt, hiddenSize,trainFcn);
    case 5
        if ~isempty(find(u <= 0, 1))
            error 'the inputs must be positive'
        end
        if ~isempty(find(y <= 0, 1))
            error 'the outputs must be positive'
        end
        ut = log(u);
        yt = log(y);
        net = convexnet(ut, yt, hiddenSize,trainFcn);
        net.trainParam = trainPar;
end

[net,~,~,~] = train(net, ut, yt);

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
    case 4
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
        
        varargout{4} = @(x) exp(net(log(x)));
end



end