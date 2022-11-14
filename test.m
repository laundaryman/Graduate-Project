%% Logistic Regression

%%%%%%% Hyperparameters %%%%%%%

N = 500;
M = 100;
rlist = [3,4,5];
eps = 1e-12;
learning_rate = 1e-3;
iternum = 200;
color = [[1,0,0];[0,1,0];[0,0,1];[0,0,0]];
al = [0,0,0,0];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A = randn(N, M);
xr = randn(M,1)*0.1;
yr = zeros(N,1);
ycand = [0,1];
for i=1:N

    prob = [1/(1+exp(A(i,:)*xr)),1-1/(1+exp(A(i,:)*xr))];
    yr(i,1) = randsample(RandStream('mlfg6331_64'), ycand, 1, true, prob);

end
init_x0 = zeros(M,1);

subplot(3,2,5);

for i=1:length(al)

    for j=1:iternum+200
        rval(j) = rlist(1)+al(i)*j;
    end

    [minY, Y] = Optimize_Nesterov("Logistic",A,b,init_x0,0,iternum+200,learning_rate,3);
    semilogy(1:1:iternum,Y(1:iternum)-minY+eps, color = color(i,:)');
    hold on;

end

t = title('(e) Logistic Regression', 'Units', 'normalized', 'Position', [0.5, -plotpos, 0]);
legend('a=1','a=0.1','a=0.01','Vanila');
xlabel('iterations');
ylabel('f-f*');