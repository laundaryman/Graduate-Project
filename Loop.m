[X,Y] = meshgrid(-6:0.3:6,-6:0.3:6);
U = zeros(size(X));
V = zeros(size(Y));


r = 2;
u = 0.19;
C = 1;

for i=1:size(X,1)
    for j=1:size(X,2)
    
        if X(i,j)*X(i,j)+Y(i,j)*Y(i,j)>r 
            U(i,j) = -C*X(i,j)/(X(i,j)^2+Y(i,j)^2);
            V(i,j) = -u*abs(Y(i,j))/Y(i,j)-C*Y(i,j)/(X(i,j)^2+Y(i,j)^2);
        end

    end
end

quiver(X,Y,U,V,0);
hold on;
rectangle('Position',[-r/2 -r/2 r r],'Curvature',[1 1])
hold on;

y0 = [2,4];
gr = [0,0];
sgr = [0,0];
lr = 1e-3;

for i=1:1000

    if y0(1,1)^2+y0(1,2)^2>r
        
        gr(1,1) = -C*y0(1,1)/(y0(1,1)^2+y0(1,2)^2);
        gr(1,2) = -C*y0(1,2)/(y0(1,1)^2+y0(1,2)^2)-u*abs(y0(1,2))/y0(1,2);

    else

        gr(1,1) = -y0(1,1)/r;
        gr(1,2) = -y0(1,2)/r;
    
    end
    
    sgr = sgr+gr;
    y0 = y0+lr*gr+lr*sgr;
    plot(y0(1,1),y0(1,2),'.',color = 'r');
    hold on;

    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);

    if i==1
        imwrite(imind, cm, 'Trajectory_loop.gif','LoopCount',inf)
    else
        imwrite(imind, cm, 'Trajectory_loop.gif','Delaytime',0,'WriteMode','append')
    end

end