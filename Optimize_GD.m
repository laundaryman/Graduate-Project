%%
%Gradient Descent Optimizer

function [minY,Y] = Optimize_GD(Problem, A, b, init_x0, lambda, iternum, learning_rate)
%% Initialization

minY = 1000000;
Y = zeros(iternum,1);
x0 = init_x0;

%% Iteration

for i=1:iternum

    grad = GetGrad(Problem, A, b, lambda, x0, learning_rate);
    x0 = x0-grad*learning_rate;
    Y(i,1) = GetFunc(Problem, A, b, lambda, x0);
    minY = min(minY, Y(i,1));  

end

