%%
%Nesterov Accelerated GD + Speed restart optimizer

function [minY,Y,R,V,Ag,Abs] = Optimize_SpeedRestart(Problem, A, b, init_x0, lambda, iternum, learning_rate, r, kmin)
%% Initialization

minY = 1000000;
Y = zeros(iternum,1);
R = zeros(iternum,1);
V = zeros(iternum,1);
Ag = zeros(iternum,1);
Abs = zeros(iternum,1);
x0 = init_x0;
x1 = x0;
y0 = x0;

%% Iteration

step = 1;
for i=1:iternum
    
    
    grad = GetGrad(Problem, A, b, lambda, y0, learning_rate);

    prvspeed = norm(x0-x1);
    x1 = x0;
    x0 = y0-grad*learning_rate;
    y0 = x0+(step-1)/(step+r-1)*(x0-x1);
    curspeed = norm(x0-x1);

    R(i,1) = 1-3/(step+2);
    if curspeed<prvspeed && step>=kmin
        step = 0;
    end

    Y(i,1) = GetFunc(Problem, A, b, lambda, x0);
    minY = min(minY, Y(i,1));
    step = step+1;

end
