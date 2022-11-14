%%
%Nesterov Accelerated GD optimizer

function [minY,Y] = Optimize_Nesterov(Problem, A, b, init_x0, lambda, iternum, learning_rate, r)
%% Initialization

minY = 1000000;
Y = zeros(iternum,1);
x0 = init_x0;
x1 = x0;
y0 = x0;

%% Iteration

for i=1:iternum
    
    grad = GetGrad(Problem, A, b, lambda, y0, learning_rate);
    x0 = y0-grad*learning_rate;
    y0 = x0+(i-1)/(i+r-1)*(x0-x1);
    x1 = x0;
    Y(i,1) = GetFunc(Problem, A, b, lambda, x0);
    minY = min(minY, Y(i,1));

end
