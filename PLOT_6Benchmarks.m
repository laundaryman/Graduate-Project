plotpos = 0.35

%% LASSO FAT

%%%%%%% Hyperparameters %%%%%%%

N = 500;
M = 100;
rlist = [3,4,5];
eps = 1e-12;
learning_rate = 1e-3;
iternum = 2500;
color = [[1,0,0];[0,1,0];[0,0,1]];
lambda = 4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A = randn(M,N);
b = randn(M,1)*5;
init_x0 = zeros(N,1);

subplot(3,2,1);

for i=1:3

    [minY, Y] = Optimize_Nesterov("LASSO",A,b,init_x0,lambda,iternum+200,learning_rate,rlist(i));
    semilogy(1:1:iternum,Y(1:iternum)-minY+eps, color = color(i,:)');
    hold on;

end

t = title('(a) Lasso with fat design', 'Units', 'normalized', 'Position', [0.5, -plotpos, 0]);
legend('r=3','r=4','r=5');
xlabel('iterations');
ylabel('f-f*');

%% LASSO SQUARE

%%%%%%% Hyperparameters %%%%%%%

N = 500;
M = 500;
rlist = [3,4,5];
eps = 1e-12;
learning_rate = 0.2*1e-4;
iternum = 3000;
color = [[1,0,0];[0,1,0];[0,0,1]];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = randn(M,N);
b = randn(M,1)*3;
init_x0 = zeros(N,1);

subplot(3,2,2);


for i=1:3

    [minY, Y] = Optimize_Nesterov("LASSO",A,b,init_x0,lambda,iternum+200,learning_rate,rlist(i));
    semilogy(1:1:iternum,Y(1:iternum)-minY+eps, color = color(i,:)');
    hold on;

end

t = title('(b) Lasso with square design', 'Units', 'normalized', 'Position', [0.5, -plotpos, 0]);
legend('r=3','r=4','r=5');
xlabel('iterations');
ylabel('f-f*');


%% NNLS FAT

%%%%%%% Hyperparameters %%%%%%%

N = 500;
M = 100;
rlist = [3,4,5];
eps = 1e-12;
learning_rate = 1e-4;
iternum = 200;
color = [[1,0,0];[0,1,0];[0,0,1]];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = randn(M,N);
b = randn(M,1)*5;
init_x0 = zeros(N,1);

subplot(3,2,3);

for i=1:3

    [minY, Y] = Optimize_Nesterov("NNLS",A,b,init_x0,lambda,iternum+200,learning_rate,rlist(i));
    semilogy(1:1:iternum,Y(1:iternum)-minY+eps, color = color(i,:)');
    hold on;

end

t = title('(c) NLS with fat design', 'Units', 'normalized', 'Position', [0.5, -plotpos, 0]);
legend('r=3','r=4','r=5');
xlabel('iterations');
ylabel('f-f*');

%% NNLS SQUARE

%%%%%%% Hyperparameters %%%%%%%

N = 500;
M = 500;
rlist = [3,4,5];
eps = 1e-12;
learning_rate = 0.2*1e-4;
iternum = 1000;
color = [[1,0,0];[0,1,0];[0,0,1]];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = randn(M,N);
b = randn(M,1)*5;
init_x0 = zeros(N,1);

subplot(3,2,4);

for i=1:3

    [minY, Y] = Optimize_Nesterov("NNLS",A,b,init_x0,lambda,iternum+200,learning_rate,rlist(i));
    semilogy(1:1:iternum,Y(1:iternum)-minY+eps, color = color(i,:)');
    hold on;

end

t = title('(d) NLS with sparse design', 'Units', 'normalized', 'Position', [0.5, -plotpos, 0]);
legend('r=3','r=4','r=5');
xlabel('iterations');
ylabel('f-f*');

%% Logistic Regression

%%%%%%% Hyperparameters %%%%%%%

N = 500;
M = 100;
rlist = [3,4,5];
eps = 1e-12;
learning_rate = 1e-3;
iternum = 200;
color = [[1,0,0];[0,1,0];[0,0,1]];

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

for i=1:3

    [minY, Y] = Optimize_Nesterov("Logistic",A,yr,init_x0,0,iternum+100,learning_rate,rlist(i));
    semilogy(1:1:iternum,Y(1:iternum)-minY+eps, color = color(i,:)');
    hold on;

end

t = title('(e) Logistic Regression with square design', 'Units', 'normalized', 'Position', [0.5, -plotpos, 0]);
legend('r=3','r=4','r=5');
xlabel('iterations');
ylabel('f-f*');

%% Logistic Regression l1reg

%%%%%%% Hyperparameters %%%%%%%

N = 500;
M = 100;
rlist = [3,4,5];
eps = 1e-12;
learning_rate = 1e-3;
iternum = 1000;
color = [[1,0,0];[0,1,0];[0,0,1]];
lambda = 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A = randn(N, M);
xr = randn(M,1)*15;
yr = zeros(N,1);
ycand = [0,1];
for i=1:N

    prob = [1/(1+exp(A(i,:)*xr)),1-1/(1+exp(A(i,:)*xr))];
    yr(i,1) = randsample(RandStream('mlfg6331_64'), ycand, 1, true, prob);

end
init_x0 = zeros(M,1);

subplot(3,2,6);

for i=1:3

    [minY, Y] = Optimize_Nesterov("Logistic l1",A,yr,init_x0,lambda,iternum+100,learning_rate,rlist(i));
    semilogy(1:1:iternum,Y(1:iternum)-minY+eps, color = color(i,:)');
    hold on;

end
t = title('(f) Logistic regression with l1 regularization', 'Units', 'normalized', 'Position', [0.5, -plotpos, 0]);
legend('r=3','r=4','r=5');
xlabel('iterations');
ylabel('f-f*');



sgtitle("Nesterov's algorithm over various benchmarks");
