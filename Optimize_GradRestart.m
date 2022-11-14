%%
%Nesterov Accelerated GD + Speed restart optimizer

function [minY,Y,R,Ag,Abs] = Optimize_GradRestart(Problem, A, b, init_x0, lambda, iternum, learning_rate, r)
%% Initialization

minY = 1000000;
Y = zeros(iternum,1);
R = zeros(iternum,1);
Ag = zeros(iternum,1);
Abs = zeros(iternum,1);
x0 = init_x0;
x1 = x0;
y0 = x0;

%% Iteration

step = 1;
for i=1:iternum

    grad = GetGrad(Problem, A, b, lambda, y0, learning_rate);
    x1 = x0;
    x0 = y0-grad*learning_rate;
    y0 = x0+(step-1)/(step+r-1)*(x0-x1);
    R(i,1) = 1-r/(step+r-1);

    if sum(sum(grad.*(x1-x0)))<0
        step = 0;
    end
    
    Y(i,1) = GetFunc(Problem, A, b, lambda, x0);
    minY = min(minY, Y(i,1));
    step = step+1;

end
