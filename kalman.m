clear all;
close all;
clc;
load('H:\windows\bureau\TP01_kalman\donnees\etat_cache.mat','x');
n = size(x,2);


load('H:\windows\bureau\TP01_kalman\donnees\observation.mat','y');

%% parametres 
c = 0.9;
alpha = -pi/8;
e = 0.1;
H = [[1 0];[0 1]] ;
R = [[cos(alpha) -sin(alpha)];[sin(alpha) cos(alpha)]] ;


%% Les variables 
xtilde= zeros(2,100);
k = zeros(2,2,100);
P_ = zeros(2,2,100);
Ptilde = zeros(2,2,100);
x_ = zeros(2,100);
w = randn(2,100);


x_(:,1) = [0;0];
P_(:,:,1) = [1 0;0 1];
Qv = [1 0;0 1];
 

%% filtre de Kalman
for i=2:n
    
    x_(:,i) = c*R*x(:,(i-1))+sqrt(1-c^2)*w(:,i);
    
    P_(:,:,i) = c*R*P_(:,:,i-1)*(c*R)'+e*e*Qv;
    
    %Gain de kalman
    k(:,:,i) = P_(:,:,i)*H'/((H*P_(:,:,i)*H'+ e*e*Qv));
    
    xtilde(:,i) = x_(:,i) + k(:,:,i)*(y(:,i)-(H*x_(:,i)));
    
    Ptilde(:,:,i) = (eye(2)-k(:,:,i)*H)/P_(:,:,i);
end 
%% Mean square error
sum = 0;
for i=1:n
    sum = sum + ((xtilde(:,i)-x(:,i)).^2);
end
error = sqrt(sum)
%% Visualisation 
figure(1);
plot(x(1,:),x(2,:),'-b');
hold on
plot(xtilde(1,:),xtilde(2,:),'-r');
axis_store =[xlim ylim];
axis square;
hold off;

%% l'etat cach� du vecteur X
figure(2)
 for k=1:n
 plot(x(1,1:k),x(2,1:k),'-b');
 hold on
 %plot(xtilde(1,1:k),xtilde(2,1:k),'-r');
 hold on;
 ellipse_conf([x(1,1:k),x(2,1:k)],[P_(1,1:k),P_(2,1:k)])
 plot(x(1,k),x(2,k),'.b');
 %plot(xtilde(1,k),xtilde(2,k),'.r');
 axis(axis_store);
 axis square;
 drawnow;
 pause(0.1);
 hold off;
 end
 
 %% Visualisation en temporel
 figure(3)
 plot(x(1,:));
 hold on;
 plot(xtilde(1,:),'r');
hold off;

figure(4)
 plot(x(2,:));
 hold on;
 plot(xtilde(2,:),'r');
hold off;
 