% This is an implementation of our ACC paper "Reduced-order
% Aggregate Dynamical Model for Wind Farms",arXiv, S.
% S. Vijayshankar and V. Purba and P. Seiler and S.
% Dhople for details of the model.

% In this code, we compare the responses of Full-order and Reduced order
% models.

load xend
clf;
N = 1
x0 = [xend(1:15)]; % Initialization
T = 50;
tspan = [0 T];

%% Scaling matrix \psi:

I = [N; N; 1; 1; N*ones(4,1) ;...
        1;1;1; N*ones(4,1); 1;1; N*ones(6,1); 1;1;1;1]; 
psi = diag(I);

%% Simulation:

fhInd = @(t,x) TurbineAggregate(t,x,1); % 1 turbine
fhROM = @(t,x) TurbineAggregate(t,x,N); % N turbines
[tInd,xInd] = ode45(fhInd,tspan,x0); 
%[tROM,xROM] = ode45(fhROM,tspan,psi*x0);
xInd(:,12) = [];


X1 = xInd(1:end-1,:);
X2 = xInd(2:end,:);


[u,s,v] = svd(X1.','econ');
figure(2)
plot(diag(s)/sum(diag(s)));


r = 4;
ur = u(:,1:r);
sr = s(1:r,1:r);
vr = v(:,1:4);

Ared = ur'*X2'*vr*inv(sr);
%% Compute power output of the wind farm:

vgd = 0;
vgq = 1;
ids = xInd(:,2);
iqs = xInd(:,1);
% igd = xROM(:,14);
% igq = xROM(:,15);

P = vgd*ids + vgq*iqs;
Q = -vgq*ids + vgd*iqs;

%% Plots

% Compare the Full and Reduced-order models
figure(1)

h2 = plot(tInd, xInd(:,12),'r');
ylabel('State')
xlabel('Time (s)')


