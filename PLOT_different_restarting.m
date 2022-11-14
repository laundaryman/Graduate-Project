%%%%%%% Hyperparameters %%%%%%%

N = 5000;
M = 50000;
rlist = [3];
eps = 1e-12;
errlist = [1e-8;1e-10;1e-12];
learning_rate = 1e-2;
iternum = 800;
color = [[1,0,0];[0,1,0];[0,0,1];[1,0,1];[0,1,1];[1,0.6,0];[1,0.3,0.6]];
Problem = "LASSO_SPARSE";
lambda = 5;
EXPNUM = 1;
AVG_NEST = zeros(size(errlist));
AVG_GR = zeros(size(errlist));
AVG_NO = zeros(size(errlist));
AVG_SR = zeros(size(errlist));
AVG_SM = zeros(size(errlist));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

BADEXP = 0;
for t=1:EXPNUM

    %%%%%%%%% Initialization %%%%%%%%%

    if Problem == "LASSO" || Problem == "NNLS" || Problem == "Logsumexp" || Problem == "SLOPE" || Problem == "LASSO_SPARSE"
    
        A = randn(N,M);
        b = randn(N,1)*3;
        init_x0 = randn(M,1);
        if Problem == "LASSO_SPARSE"
            A = sprandn(N, M, 0.005)*0.2;
            x0 = randn(M,1);
            z = randn(N,1);
            b = A*x0+z;
        end
        if Problem=="SLOPE"
            b = A*rand(N,1)+randn(M,1);
        end
    
    elseif Problem == "Logistic" || Problem == "Logistic l1" || Problem=="Logistic_SPARSE"
    
          tic

        if Problem == "Logistic_SPARSE"
            A = sprand(N, M, 0.001);
        else
            A = randn(N, M);
        end
        xr = randn(M,1)*0.1;
        b = zeros(N,1);
        ycand = [0,1];
        for i=1:N
        
            prob = [1/(1+exp(A(i,:)*xr)),1-1/(1+exp(A(i,:)*xr))];
            b(i,1) = randsample(RandStream('mlfg6331_64'), ycand, 1, true, prob);
        
        end

        init_x0 = randn(M,1);
    
    elseif Problem == "Matrix Completion"

        diagm = zeros(N,1);
        for i=1:5
            diagm(i,1) = i;
        end
        M = RandUGroup(N,'O')*diag(diagm)*RandUGroup(N,'O')';
        A = zeros(N,N);
        b = zeros(N,N);
        for i=1:N
            for j=1:N
                if rand(1)<0.1
                    A(i,j) = M(i,j);
                    b(i,j) = 1;
                end
            end
        end
        init_x0 = randn(N,N);
    
    end

    fprintf("----------EXPERIMENT %d ON-----------\n",t);
  
    minY = ones(6, 1)*1000000;
    CV_NEST = ones(size(errlist))*(iternum+200);
    CV_SR = ones(size(errlist))*(iternum+200);
    CV_GR = ones(size(errlist))*(iternum+200);
    CV_NO = ones(size(errlist))*(iternum+200);
    CV_SM = ones(size(errlist))*(iternum+200);

    Y = zeros(6, iternum+200);
    [minY(1), Y(1,:)] = Optimize_Nesterov(Problem,A,b,init_x0,lambda,iternum+200,learning_rate,rlist(1));
    [minY(2), Y(2,:), R2, V2, A2, Abs2] = Optimize_SpeedRestart(Problem,A,b,init_x0,lambda,iternum+200,learning_rate,rlist(1),10);
    [minY(3), Y(3,:), R3,A3,Abs3] = Optimize_GradRestart(Problem,A,b,init_x0,lambda,iternum+200,learning_rate,rlist(1));
    [minY(4), Y(4,:)] = Optimize_GD(Problem,A,b,init_x0,lambda,iternum+200,learning_rate); 
    [minY(5), Y(5,:), R5, xopt,xrt,xpv] = Optimize_NewOpt(Problem,A,b,init_x0,lambda,iternum+200,learning_rate,1);
   % [minY(6), Y(6,:), R6] = Optimize_NewOpt(Problem,A,b,init_x0,lambda,iternum+200,learning_rate,1,2);
    toc
    
    for j=1:3
        for i=1:iternum+200
            if (Y(1,i)-min(minY))<errlist(j,1)
                CV_NEST(j,1) = i;
                break;
            end
        end
        for i=1:iternum+200
            if (Y(2,i)-min(minY))<errlist(j,1)
                CV_SR(j,1) = i;
                break;
            end
        end
        for i=1:iternum+200
            if (Y(3,i)-min(minY))<errlist(j,1)
                CV_GR(j,1) = i;
                break;
            end
        end
        for i=1:iternum+200
            if (Y(5,i)-min(minY))<errlist(j,1)
                CV_NO(j,1) = i;
                break;
            end
        end
        for i=1:iternum+200
            if (Y(6,i)-min(minY))<errlist(j,1)
                CV_SM(j,1) = i;
                break;
            end
        end
    end

    if CV_NEST(3,1)*CV_SR(3,1)*CV_GR(3,1)*CV_NO(3,1) == 0
        BADEXP = BADEXP+1;
    else

        for j=1:3
            AVG_NEST(j,1) = AVG_NEST(j,1)+CV_NEST(j,1);
            AVG_SR(j,1) = AVG_SR(j,1)+CV_SR(j,1);
            AVG_GR(j,1) = AVG_GR(j,1)+CV_GR(j,1);
            AVG_NO(j,1) = AVG_NO(j,1)+CV_NO(j,1);
            AVG_SM(j,1) = AVG_SM(j,1)+CV_SM(j,1);
        end

    end

end

min(minY)
for i=1:6

    semilogy(1:1:iternum,Y(i,1:iternum)-min(minY)+eps, color = color(i,:)');
    hold on;

end


legend('Nesterov','Speed restart','Gradient restart','Gradient Descent', 'New algorithm');
title(Problem);
xlabel('iterations');
ylabel('f-f*');



figure;

plot(1:1:iternum, R2(1:iternum), color = 'g');
hold on
plot(1:1:iternum, R3(1:iternum), color = 'b');
hold on
plot(1:1:iternum, R5(1:iternum), color = 'c');



BADEXP
fprintf("-----------NESTEROV--------------")
AVG_NEST/(EXPNUM-BADEXP)
fprintf("-----------SR--------------")
AVG_SR/(EXPNUM-BADEXP)
fprintf("-----------GR--------------")
AVG_GR/(EXPNUM-BADEXP)
fprintf("-----------NA--------------")
AVG_NO/(EXPNUM-BADEXP)