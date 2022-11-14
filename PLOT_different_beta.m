%%%%%%% Hyperparameters %%%%%%%

N = 500;
M = 500;
rlist = [3,4,5];
eps = 1e-12;
learning_rate = 3e-4;
iternum = 200;
lambda = 4;
errlist = [1e-8,1e-10,1e-12];
EXPNUM = 1;
Problem = "NNLS";


%% Plot for different beta: number of steps needed

blist = 0.9:0.02:1.2;
ERR1 = zeros(3,EXPNUM);
ERRMIN = ones(3,EXPNUM)*iternum;

for i=1:EXPNUM

    %%%%%%%%%% Initialization %%%%%%%%%%%

    if Problem == "LASSO" || Problem == "NNLS"
    
        A = randn(M,N);
        b = randn(M,1)*2;
        init_x0 = randn(N,1)*100;
        learning_rate = 1/max(eig(A'*A))
    
    else
    
        A = randn(N, M);
        xr = randn(M,1)*0.1;
        b = zeros(N,1);
        ycand = [0,1];
        for u=1:N
        
            prob = [1/(1+exp(A(u,:)*xr)),1-1/(1+exp(A(u,:)*xr))];
            b(u,1) = randsample(RandStream('mlfg6331_64'), ycand, 1, true, prob);
        
        end
        init_x0 = zeros(M,1);
    end

    fprintf("----------EXPERIMENT %d ON-----------\n",i);
    for j=1:length(blist)

        [minY, Y, R4] = Optimize_NewOpt(Problem,A,b,init_x0,lambda,iternum+200,learning_rate,blist(j),1);
        for k=1:3
            ad = iternum;
            for t=1:iternum
                if (Y(t,1)-minY)<errlist(1,k)
                    ad = t;
                    break;
                end
            end
            if blist(j) == 1
                ERR1(k,i) = ad;
            end
            ERRMIN(k,i) = min(ERRMIN(k,i),ad);
        end

    end

end

for k=1:3
    figure;
    plot(1:1:EXPNUM,ERR1(k,:),'.','MarkerSize',15,color = 'r');
    hold on;
    plot(1:1:EXPNUM,ERRMIN(k,:),'.','MarkerSize',15,color = 'b');
    hold on;
end


%% Different beta plot

blist = [0.5,0.9,0.99,1,1.01,1.1,1.3];
c = ['r','g','b','c','m','y','k'];

figure;

for j=1:7

    [minY, Y, A1] = Optimize_NewOpt(Problem,A,b,init_x0,lambda,iternum+200,learning_rate,blist(j),0);
    semilogy(1:1:iternum,Y(1:iternum)-minY+1e-14,color = c(j));
    hold on;

end

