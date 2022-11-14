%%
%Adaptive Nesterov Optimizer

function [minY,Y] = Optimize_ADANest(Problem, A, b, init_x0, lambda, iternum, rt, learning_rate)
%% Initialization

minY = 10000;
Y = zeros(iternum);
x0 = init_x0;
y0 = x0;
x1 = x0;

%% Iteration

for i=1:iternum

    grad = GetGrad(Problem, A, b, lambda, y0, learning_rate);
    [x1,x0,y0] = Nesterov([x1 x0 y0],grad,learning_rate, rt(i), i);
    Y(i) = GetFunc(Problem, A, b, lambda, x0);
    minY = min(minY, Y(i));

end
