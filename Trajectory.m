iteration = 100000;
lr = 1e-6;
xs = 3;
ys = 10;

yx = 3;
yy = 10;


for i=1:iteration

    prx = xs;
    pry = ys;
    xs = yx-lr*yy*exp(-2*atan(yx/yy))*(exp(atan(yx/yy))+2)/(yx^2+yy^2);
    ys = yy+lr*yx*exp(-2*atan(yx/yy))*(exp(atan(yx/yy))+2)/(yx^2+yy^2);

    yx = xs+(xs-prx);
    yy = ys+(ys-pry);
    plot(xs,ys,'.',color = 'r');
    hold on;

end

%{
time = 0:0.001:15;
x = (1+10./exp(time)).*sin(time);
y = (1+10./exp(time)).*cos(time);

plot(x,y);
figure;
plot(time,x.^2+y.^2);
%}