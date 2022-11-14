% Generate uniform randomly p Orhogonal/Hermitian matrix in R^n 
n               = 3;      % space dimension
p               = 1e6;    % number of (n x n) matrices
GType           = 'O';   % among 'O', 'SO', 'U', 'SU'

U = RandUGroup(n, 1, p, GType);

% Check for n=2 or 3
U1 = reshape(U(:,1,:),[n p]);
if n==2
    x = real(U1(1,:));
    y = real(U1(2,:));
    tt = atan2(y,x);
    
    close all
    subplot(1,2,1)
    histogram(tt,100)
    subplot(1,2,2)
    plot(x,y,'.','markersize',0.1)
    axis equal
elseif n==3
    x = real(U1(1,:));
    y = real(U1(2,:));
    z = real(U1(3,:));
    
    [az,el,~] = cart2sph(x,y,z);
    azi = linspace(-pi,pi,65);
    eli = asin(linspace(-1,1,33));
    close all
    subplot(1,2,1)
    histogram2(az,el,azi,eli,'FaceColor','c');
    subplot(1,2,2)
    plot3(x,y,z,'.','markersize',0.001)
    axis equal  
end
