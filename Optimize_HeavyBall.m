%%
%Heavy-Ball Method Optimizer

function [minY,Y] = Optimize_HeavyBall(Problem, A, b, init_x0, lambda, iternum, u, learning_rate)
%% Initialization

minY = 10000;
Y = zeros(iternum,1);
x0 = init_x0;
x1 = x0;
y0 = x0;

%% Iteration

for i=1:iternum

    grad = GetGrad(Problem, A, b, lambda, y0, learning_rate);
    [x1,x0,y0] = HeavyBall([x1 x0 y0],grad,learning_rate, u);
    Y(i,1) = GetFunc(Problem, A, b, lambda, x0);
    minY = min(minY, Y(i,1));

end
