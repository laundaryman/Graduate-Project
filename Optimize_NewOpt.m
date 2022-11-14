%%
%New speed restart optimizer that is almost identical to gradient restart

function [minY,Y,R,x0,xrt,xpv] = Optimize_NewOpt(Problem, A, b, init_x0, lambda, iternum, learning_rate, r)
%% Initialization

minY = 1000000000;
Y = zeros(iternum,1);
R = zeros(iternum,1);
V = zeros(iternum,1);
Ag = zeros(iternum,1);
Abs = zeros(iternum,1);
x0 = init_x0;
x1 = x0;
y0 = x0;
xrt = x0;
xpv = x0;
chk = 0;

%% Nesterov algorithm + beta=1

for i=1:iternum

    grad = GetGrad(Problem, A, b, lambda, y0, learning_rate);
    dotp = sum(sum(grad.*(x1-x0)));

    if dotp<0
        
        x1 = x0;
        x0 = x1 - learning_rate*GetGrad(Problem, A, b, lambda, x1, learning_rate);
        y0 = x0;
        R(i,1) = 0;

    else
   
        x1 = x0;
        x0 = y0-learning_rate*grad;
        y0 = x0+r*(x0-x1);
        R(i,1) = 1;

    end

    Y(i,1) = GetFunc(Problem, A, b, lambda, x0);
    if Y(i,1)>1e18 && chk == 0
        xrt = x0;
        xpv = x1;
        chk = 1
    end
    minY = min(minY, Y(i,1));

end
%% Order 3 algorithm
%{
for i=1:iternum

    while 1
        grad = GetGrad(Problem, A, b, lambda, x0, learning_rate);
        if GetFunc(Problem, A, b, lambda, x0-grad*learning_rate)<GetFunc(Problem, A, b, lambda, x0)-0.5*learning_rate*norm(grad)^2;
            break;
        elseif norm(grad)^2<1e-8
            break;
        else
            learning_rate = learning_rate/2;
        end
    end

    grad = GetGrad(Problem, A, b, lambda, y0, learning_rate);
    if sum(sum(grad.*(x1-x0)))<0
        x2 = x0;
        x1 = x0;
        x0 = x0 - learning_rate*GetGrad(Problem, A, b, lambda, x0, learning_rate);
        y0 = x0;
        R(i,1) = 0;
    else
        x2 = x1;
        x1 = x0;
        x0 = y0-learning_rate*grad;
        y0 = x0+r*(x0-x1)+er*(x0-2*x1+x2);
        R(i,1) = 1;
    end

    Y(i,1) = GetFunc(Problem, A, b, lambda, x0);
    minY = min(minY, Y(i,1));

end
%}

