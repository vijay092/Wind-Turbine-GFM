function [ F ] = TurbineAggregate(t,x,N)

% This function lists out differential equations governing the turbine.

% Initialize d/dts to 0.
F = zeros(size(x));

%% Turbine states
% Rename states to ease debugging



iqs = x(1);
ids = x(2);
eqs = x(3);
eds = x(4);
phiTe = x(5);
phiIq = x(6);
phiQs = x(7);
phiId = x(8);
wr = x(9);
theta_tw = x(10); 
wt = x(11);
theta = x(12);
E_star = x(13);
Igd = x(14);
Igq = x(15);

% DFIG Params
wnom = 2*pi*60; %
ws = 1;
Lm = 4;
Lss = 1.01*Lm;
Lrr = 1.008*Lss;
Kmrr = Lm/Lrr;
Ls = Lss - Lm*Kmrr;
Rs = 0.005;
Rr = 1.1*Rs;
R2 = Kmrr^2 * Rr;
R1 = Rs + R2;
Tr = 10;
Xm = 2*pi*60*Lm;

% Stator voltages
vdg = 0;
vqg = 1;

% Mechanical params
Kopt = 0.5787;
Qsref = 0.01;
Teref = Kopt*wr^2*N ;
rho = 1.225;
R_turb = 58.6;
vw = 12;
Prated = 5e6;
GB = 145.5;
Tmbase =  GB * Prated * 1/(wnom/2);
wtB = wnom/(2*GB);
lambda = wt* R_turb /vw;
c1 = 0.22; c2=116; c3 = 0.4; c6= 5; c7=12.5; c8=0.08; c9=0.035; c10=0;
Cp = c1*( (c2/lambda) - c2*c9 -c6 ) * exp(-c7/lambda) + c10*lambda;
Ht = 4; Hg = 0.1*Ht; ksh = 0.3; csh = 0.01;

% Control gains of the rotor-side controller
Kiq = -1;
Kte = -1.5;
Kid = -0.5;
Kqs = 1;
Tiq = 0.0025;
Tte = 0.025;
Tid = 0.005;
Tqs = 0.05;

% Parametric scaling:
R1 = R1/N;
Ls = Ls/N;
Kid  = Kid/N;
Kiq = Kiq /N;
Xm = Xm/N;
R2 = R2/N;
ksh = ksh*N;
csh = csh*N;
Ht = Ht*N;
Hg = Hg*N;
Tmbase = Tmbase/N;

% Some intermediate variables:

Tm = 0.5 * rho * pi* R_turb^2 * Cp * vw^3 /(wt*Tmbase); % Eq. 4
Te =  +  (eqs /ws) * iqs + (eds /ws) * ids; % T_e (Electrical Torque)

qr =  -vqg * ids + vdg * iqs; % Eq. 14

iqr  = - eds /Xm - Kmrr * iqs; % Eq. 13
idr = + eqs /Xm - Kmrr * ids; % Eq. 13

vqr =  Kiq* Kte * (Teref - Te) + Kiq * Kte /Tte * phiTe - Kiq * iqr + Kiq/Tiq * phiIq; % vqr
vdr = +  Kid* Kqs * (Qsref - qr) + Kid * Kqs /Tqs * phiQs - Kid * idr + Kid/Tid * phiId; % vdr

% Differential Equations governing the rectifier/ controllers
F(1) = wnom / Ls * ( -R1 * iqs + ws * Ls * ids + wr/ws * eqs - 1/(Tr * ws) * eds - vqg + Kmrr * vqr); % Eq. 5
F(2) = wnom / Ls * ( -R1 * ids - ws * Ls * iqs + wr/ws * eds - 1/(Tr * ws) * eqs - vdg + Kmrr * vdr); % Eq. 5

F(3) = wnom * ws * ( R2 * ids  + (1 - wr/ws) * eds - 1/(Tr * ws) * eqs  - Kmrr * vdr); % Eq. 5
F(4) = wnom * ws * (- R2 * iqs  - (1 - wr/ws) * eqs - 1/(Tr * ws) * eds  + Kmrr * vqr); % Eq. 5

F(5) = Teref - Te; % Eq. 10
F(6) = Kte * (Teref - Te)  + Kte /Tte * phiTe  - iqr; % Eq. 10

F(7) = Qsref - qr; % Eq. 9
F(8) = Kqs * (Qsref - qr)  + Kqs /Tqs * phiQs  - idr; % Eq. 9

% Aerodynamic model:
F(9) = 1/(2*Hg) * (ksh*theta_tw + csh*wnom* (wt - wr) - Te); % Eq. 1
F(10) = wnom*(wt - wr); % Eq. 2
F(11) = 1/(2*Ht) * (Tm - ksh * theta_tw - csh*wnom*(wt - wr)); % Eq. 3

vgd = 0; vgq = 1;
P_star = vgd*ids + vgq*iqs; % output of the turbine
Q_star = -vgq*ids + vgd*iqs; % output of the turbine


% VOC
old_base_value_volt = 208;
new_base_value_volt = 630;
old_base_value_VA = 1500;
new_base_value_VA = 5e6;
old_base_value_F = old_base_value_volt^2/old_base_value_VA;
new_base_value_F = new_base_value_volt^2/new_base_value_VA;


wb = 2*pi*60;

k1 = 0.0033 * old_base_value_F/new_base_value_F;
e1 = [1 0]'; e2 = [0 1]';
psi = pi/4;
S_star = [P_star;Q_star];
Eb = old_base_value_volt/new_base_value_volt;
Lg = 0.0196*old_base_value_F/new_base_value_F;
C = 0.1086*new_base_value_F/old_base_value_F;
Rg = 0.0139*old_base_value_F/new_base_value_F;
Igdq = [Igd;Igq];
VDQ = [1;0];
Kb = 0.0347*old_base_value_F/new_base_value_F;
k2 = 0.0796*new_base_value_F^2/old_base_value_F^2;



% % Compute rho by solving a nonlinear eqn:
% fun = @(rho) rho + epsilon*log(exp(-1/epsilon)...
%       + exp((-Imax*sqrt(C^2*Kb^2*(rho-1)^2 + rho^2))/(epsilon)));
% x0 = [0,0];
% x = fsolve(fun,x0)
if 1.2/norm(Igdq)^2  < 1
    rho = 1.2/norm(Igdq)^2;
    
else
    
    rho = 1;
end

P = Igdq'*(rho/C*Ti(rho)'*T2(pi/2)' - 1/C*T2(pi/2)')*Igdq...
    +rho/C*e1'*Tv(rho)'*T2(pi/2)'*E_star*Igdq;
Q = Igdq'*(1/C*eye(2) - rho/C*Ti(rho)') * Igdq -...
    rho/C * e1' *Tv(rho)'*E_star*Igdq;

S = [P;Q];


F(12) = wb + wb*k1/E_star^2* e1'*T2(psi - pi/2) * (S_star - S); % theta

F(13) = wb*k1/E_star^2 * e2'*T2(psi - pi/2) * (S_star - S)... % E_star
    + wb *k2*(Eb^2 - E_star^2)*E_star;

F(14:15) = wb * ( T2(pi/2) * ( eye(2) - 1/(Lg*C)*(eye(2) - rho * Ti(rho))) - Rg/Lg*eye(2))*Igdq...
    + wb/Lg * (rho/C * T2(pi/2) * Tv(rho) * e1 * E_star - T2(rho)*VDQ); % Igdq




    function T1 = T1(alpha)
        T1 = 2/3*[cos(alpha) cos(alpha - 2*pi/3)  cos(alpha + 2*pi/3);...
            -sin(alpha) -sin(alpha - 2*pi/3) -sin(alpha + 2*pi/3)];
    end

    function T2 = T2(alpha)
        T2 = [cos(alpha) sin(alpha);...
            -sin(alpha) cos(alpha)];
    end

    function Ti = Ti(rho)
        
        Ti = [rho/d(rho)  -C*Kb*(rho - 1)/d(rho);
            C*Kb*(rho-1)/d(rho)   rho/d(rho)];
    end

    function Tv = Tv(rho)
        Tv = [-C^2*Kb*(rho-1)/d(rho)  0;
            C*rho/d(rho)                 0];
        
    end

    function Tiv  = Tiv(rho)
        
        Tiv = [((C*Lg - 1)* Kb* rho* (rho-1) + Rg*rho^2) /dg(rho) , 0;
            (C*Rg*Kb*rho*(rho-1) - Lg*rho^2)/dg(rho), 0];
        
    end

    function Tgv = Tgv(rho)
        
        Tgv = [(Kb*rho*(rho-1) - Rg*d(rho))/dg(rho)   , -ng(rho)/dg(rho);
            ng(rho)/dg(rho)            , (Kb*rho*(rho-1) - Rg*d(rho-1))/dg(rho)];
        
    end

    function d = d(rho)
        
        d = C^2 * Kb^2 *(rho-1)^2 + rho^2;
    end

    function ng = ng(rho)
        
        ng = Lg*rho^2 + C*Kb^2*(rho-1)^2 * (C*Lg - 1)
        
    end

    function dg = dg(rho)
        
        dg = (C*Lg - 1)^2*Kb^2*(rho-1)^2 + (C*Rg)^2 * Kb^2 * (rho-1)^2 ...
            -2*Kb*Rg*rho*(rho-1) + rho^2 *(Rg^2 + Lg^2);
        
        
    end

end

