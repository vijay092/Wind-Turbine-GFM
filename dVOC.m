clearvars
format long;

Flag_InverterType = 3; % 0 for Hugo's utility params, 1 for Brian's lab params, 2 for Minghui's cal, 3 for NREL 3phase inverter parameters from Gab-Su

% InverterBus = [10,2,3,4,5,6,7,8,9];
InverterBus = [2,3,4,5,6,7,8,9,10];

p_L = 1;

%% System data

mpc = loadcase('case39');

sysdata_load = mpc.bus(1:29,:); %?
sysdata_load_P = sysdata_load(:,3);
sysdata_load_Q = sysdata_load(:,4);
sysdata_load_Ptotal = sum(sysdata_load_P);
sysdata_load_Qtotal = sum(sysdata_load_Q);

sysdata_gen = mpc.gen;
sysdata_gen_P = sysdata_gen(:,2);
sysdata_gen_Q = sysdata_gen(:,3);
sysdata_gen_Pmax = sysdata_gen(:,9);
sysdata_gen_Qmax = sysdata_gen(:,4);


mpc_new = mpc;
% mpc_new.bus(:,5) = mpc_new.bus(:,3)./(mpc_new.bus(:,8).^2);
% mpc_new.bus(:,6) = -mpc_new.bus(:,4)./(mpc_new.bus(:,8).^2);
% mpc_new.bus(:,3) = 0;
% mpc_new.bus(:,4) = 0;

% runpf(mpc)
% runpf(mpc_new)

% system base
S_base_sys = 100e6;
V_base_sys = 345e3;


%% Machine parameters

P_base_sys = S_base_sys;
% V_base_sys = V_base_sys/sqrt(3)*sqrt(2); % Machine terminal V_LN,max
V_base_sys = V_base_sys*sqrt(2);
omega_base_sys = 2*pi*60;
I_base_sys = P_base_sys/(3/2*V_base_sys); % Peak line current
Z_base_sys = V_base_sys/I_base_sys;
L_base_sys = Z_base_sys/omega_base_sys;
C_base_sys = 1/(Z_base_sys*omega_base_sys); % ?
Ifd_base_sys = 4.5696/40*I_base_sys; % Value of Lad/Lafd from Kundur example
Vfd_base_sys = P_base_sys/Ifd_base_sys;
Zfd_base_sys = Vfd_base_sys/Ifd_base_sys;
Lfd_base_sys = Zfd_base_sys/omega_base_sys;
phifd_base_sys = Vfd_base_sys/omega_base_sys;
T_base_sys = P_base_sys/omega_base_sys; % Torque

omega_nom = 1;

% Machine
H = zeros(10,1);
H(1,1) = 42;
H(2,1) = 30.3;
H(3,1) = 35.8;
H(4,1) = 28.6;
H(5,1) = 26;
H(6,1) = 34.8;
H(7,1) = 26.4;
H(8,1) = 24.3;
H(9,1) = 34.5;
H(10,1) = 500;

% M = 2*H/(120*pi);
M = 2*H;


Ra = zeros(10,1);

xdp = zeros(10,1);
xdp(1,1) = 0.031;
xdp(2,1) = 0.0697;
xdp(3,1) = 0.0531;
xdp(4,1) = 0.0436;
xdp(5,1) = 0.132;
xdp(6,1) = 0.05;
xdp(7,1) = 0.049;
xdp(8,1) = 0.057;
xdp(9,1) = 0.057;
xdp(10,1) = 0.006;

xqp = zeros(10,1);
xqp(1,1) = 0.008;
% xqp(1,1) = 0.069;
xqp(2,1) = 0.17;
xqp(3,1) = 0.0876;
xqp(4,1) = 0.166;
xqp(5,1) = 0.166;
xqp(6,1) = 0.0814;
xqp(7,1) = 0.186;
xqp(8,1) = 0.0911;
xqp(9,1) = 0.0587;
xqp(10,1) = 0.008;

xd = zeros(10,1);
xd(1,1) = 0.1;
xd(2,1) = 0.295;
xd(3,1) = 0.2495;
xd(4,1) = 0.262;
xd(5,1) = 0.67;
xd(6,1) = 0.254;
xd(7,1) = 0.295;
xd(8,1) = 0.29;
xd(9,1) = 0.2106;
xd(10,1) = 0.02;

xq = zeros(10,1);
xq(1,1) = 0.069;
xq(2,1) = 0.282;
xq(3,1) = 0.237;
xq(4,1) = 0.258;
xq(5,1) = 0.62;
xq(6,1) = 0.241;
xq(7,1) = 0.292;
xq(8,1) = 0.28;
xq(9,1) = 0.205;
xq(10,1) = 0.019;

Td0p = zeros(10,1);
Td0p(1,1) = 10.2;
Td0p(2,1) = 6.56;
Td0p(3,1) = 5.7;
Td0p(4,1) = 5.69;
Td0p(5,1) = 5.4;
Td0p(6,1) = 7.3;
Td0p(7,1) = 5.66;
Td0p(8,1) = 6.7;
Td0p(9,1) = 4.79;
Td0p(10,1) = 7;

Tq0p = zeros(10,1);
Tq0p(1,1) = 0.1; % 0 in Ian's doc
% Tq0p(1,1) = 1;
Tq0p(2,1) = 1.5;
Tq0p(3,1) = 1.5;
Tq0p(4,1) = 1.5;
Tq0p(5,1) = 0.44;
Tq0p(6,1) = 0.4;
Tq0p(7,1) = 1.5;
Tq0p(8,1) = 0.41;
Tq0p(9,1) = 1.96;
Tq0p(10,1) = 0.7;

D = zeros(10,1);

% AVR
TR = zeros(10,1);
TR(:) = 0.01;

TA = zeros(10,1);
TA(:) = 0.4;

KA = zeros(10,1);
KA(:) = 20;

TF = zeros(10,1);
TF(:) = 0.35;

KF = zeros(10,1);
KF(:) = 0.09;

TE = zeros(10,1);
TE(:) = 0.57;

% PSS
Kpss = zeros(10,1);
Kpss(1,1) = 1/(120*pi);
Kpss(2,1) = 0.5/(120*pi);
Kpss(3,1) = 0.5/(120*pi);
Kpss(4,1) = 2/(120*pi);
Kpss(5,1) = 1/(120*pi);
Kpss(6,1) = 4/(120*pi);
Kpss(7,1) = 7.5/(120*pi);
Kpss(8,1) = 2/(120*pi);
Kpss(9,1) = 2/(120*pi);
Kpss(10,1) = 1/(120*pi);
Kpss = Kpss*(120*pi);

TW = zeros(10,1);
TW(:) = 10;

T1 = zeros(10,1);
T1(1,1) = 1;
T1(2,1) = 5;
T1(3,1) = 3;
T1(4,1) = 1;
T1(5,1) = 1.5;
T1(6,1) = 0.5;
T1(7,1) = 0.2;
T1(8,1) = 1;
T1(9,1) = 1;
T1(10,1) = 5;

T2 = zeros(10,1);
T2(1,1) = 0.05;
T2(2,1) = 0.4;
T2(3,1) = 0.2;
T2(4,1) = 0.1;
T2(5,1) = 0.2;
T2(6,1) = 0.1;
T2(7,1) = 0.02;
T2(8,1) = 0.2;
T2(9,1) = 0.5;
T2(10,1) = 0.6;

T3 = zeros(10,1);
T3(1,1) = 3;
T3(2,1) = 1;
T3(3,1) = 2;
T3(4,1) = 1;
T3(5,1) = 1;
T3(6,1) = 0.5;
T3(7,1) = 0.5;
T3(8,1) = 1;
T3(9,1) = 2;
T3(10,1) = 3;

T4 = zeros(10,1);
T4(1,1) = 0.5;
T4(2,1) = 0.1;
T4(3,1) = 0.2;
T4(4,1) = 0.3;
T4(5,1) = 0.1;
T4(6,1) = 0.05;
T4(7,1) = 0.1;
T4(8,1) = 0.1;
T4(9,1) = 0.1;
T4(10,1) = 0.5;

TM = zeros(10,1);
TM(:) = 0.2;

RD = zeros(10,1);
RD(:) = 0.05;



%% Inverter parameters

% V_inv_rating = 24e3;
V_inv_rating = 345e3;

if Flag_InverterType == 0
    
    V_unscale = 208;
    S_unscale = 50e3;
    Lf_unscale_pu = 0.15;
    rf_unscale_pu = 0.0015;
    Cf_unscale_pu = 0.0326;
    Rd_unscale_pu = 0.4623;
    Lc_unscale_pu = 0.03;
    rc_unscale_pu = 0.0012;
    
    z_unscale_base = V_unscale^2/S_unscale;
    
    Lf_unscale = (z_unscale_base/omega_base_sys)*Lf_unscale_pu;
    rf_unscale = z_unscale_base*rf_unscale_pu;
    Cf_unscale = (1/z_unscale_base/omega_base_sys)*Cf_unscale_pu;
    Rd_unscale = z_unscale_base*Rd_unscale_pu;
    Lc_unscale = (z_unscale_base/omega_base_sys)*Lc_unscale_pu;
    rc_unscale = z_unscale_base*rc_unscale_pu;
    
elseif Flag_InverterType == 1
    
    V_unscale = 208;
    S_unscale = 1e3;
    z_unscale_base = V_unscale^2/S_unscale;
    
    Lf_unscale = 1*10^-3;
    rf_unscale = 0.7;
    Cf_unscale = 24*10^-6;
    Rd_unscale = 0.02;
    Lc_unscale = 0.2*10^-3;
    rc_unscale = 0.12;
    
    Lf_unscale_pu = Lf_unscale/(z_unscale_base/omega_base_sys);
    rf_unscale_pu = rf_unscale/z_unscale_base;
    Cf_unscale_pu = Cf_unscale/(1/z_unscale_base/omega_base_sys);
    Rd_unscale_pu = Rd_unscale/z_unscale_base;
    Lc_unscale_pu = Lc_unscale/(z_unscale_base/omega_base_sys);
    rc_unscale_pu = rc_unscale/z_unscale_base;
    
elseif Flag_InverterType == 2
    
    V_unscale = 208;
    S_unscale = 10e3;
    z_unscale_base = V_unscale^2/S_unscale;
    
    omega_nom_SI = 60;
    f_nom_SI = 60*2*pi;
    f_sw = 5*1e3;
    V_dc = 600;
    deltaI = 0.1*S_unscale/(3*V_unscale);
    
    Lf_unscale = V_dc/(12*f_sw*deltaI);
    rf_unscale = 0.2;
    Cf_unscale = 0.05*S_unscale/(omega_nom_SI*V_unscale^2);
    Rd_unscale = 0.02/(S_unscale/1e3);
    Lc_unscale = Lf_unscale;
    rc_unscale = rf_unscale;
    
    Lf_unscale_pu = Lf_unscale/(z_unscale_base/omega_base_sys);
    rf_unscale_pu = rf_unscale/z_unscale_base;
    Cf_unscale_pu = Cf_unscale/(1/z_unscale_base/omega_base_sys);
    Rd_unscale_pu = Rd_unscale/z_unscale_base;
    Lc_unscale_pu = Lc_unscale/(z_unscale_base/omega_base_sys);
    rc_unscale_pu = rc_unscale/z_unscale_base;
    
    % Temporary, need to update
    mu_v_unscale = 120;
    mu_i_unscale = 0.036;
    xi_unscale = 15;
    Cvoc_unscale = 0.2679;
    
elseif Flag_InverterType == 3
    % NREL Inverter 3-phase 208V 3kVA
    V_unscale = 208;
    S_unscale = 1500;
    z_unscale_base = V_unscale^2/S_unscale;
    
    %     Lf_unscale = 1*10^-3*p_L;
    %     rf_unscale = 0.433*1*p_L;
    %     Cf_unscale = 24*10^-6;
    %     Rd_unscale = 0.2;
    %     Lc_unscale = 0.2*10^-3;
    %     rc_unscale = 0.05*1;% 0.12;
    %
    %
    %     Lf_unscale_pu = Lf_unscale/(z_unscale_base/omega_base_sys);
    %     rf_unscale_pu = rf_unscale/z_unscale_base;
    %     Cf_unscale_pu = Cf_unscale/(1/z_unscale_base/omega_base_sys);
    %     Rd_unscale_pu = Rd_unscale/z_unscale_base;
    %     Lc_unscale_pu = Lc_unscale/(z_unscale_base/omega_base_sys);
    %     rc_unscale_pu = rc_unscale/z_unscale_base;
    % % VOC design for 5% droops.
    %     mu_v_unscale = 120;
    %     mu_i_unscale = 0.12;
    %     xi_unscale = 54*1; % DESIGNED for 5% V-Q droop.
    %     Cvoc_unscale = 0.053*1; % % Designed for 5% f-P droop.
    
    psi = pi/4;
    epsilon = 0.1;
    Eb = 1;
    Imax = 1.2;
    Li = 0.0196;
    Lg = 0.0196;
    C = 0.1086;
    Ri = 0.0139;
    Rg = 0.0139;
    Kb = 0.0347;
    K_Pi = 0.9817;
    K_Ii = 0.6944;
    k1 = 0.0033;
    K_Pv = 1.4476;
    K_Iv = 10.2944;
    wbw_i = 50;
    wbw_v = 13.33;
    k2 = 0.0796;
    
    
end


for i = 1
    
    P_inv_rating(i,1) = sysdata_gen_Pmax(i)*1e6;
    Q_inv_rating(i,1) = P_inv_rating(i,1)*1; % Inverter reactive power rating
    
        kappa_p = sysdata_gen_Pmax(i)*1e6/S_unscale;
        kappa_v = V_inv_rating/V_unscale;
        PIpll_base_sys = omega_base_sys/V_base_sys;
        PIp_base_sys = I_base_sys/P_base_sys;
        PIi_base_sys = V_base_sys/I_base_sys;
    
    %     % VOC parameter scaling, may need to update
    % %     xi(i,1) = xi_unscale/kappa_v^2;
    %     xi(i,1) = xi_unscale;
    %     Cvoc(i,1) = Cvoc_unscale;
    %     mu_v(i,1) = mu_v_unscale*kappa_v/V_base_sys;
    %     mu_i(i,1) = mu_i_unscale*kappa_v/kappa_p*I_base_sys;
    %
    %     % line parameters
    %     Lf_1(i,1) = Lf_unscale/kappa_p*kappa_v^2/L_base_sys;
    %     rf_1(i,1) = rf_unscale/kappa_p*kappa_v^2/Z_base_sys;
    %     Cf(i,1) = Cf_unscale*kappa_p/kappa_v^2/C_base_sys;
    %     Rd(i,1) = Rd_unscale/kappa_p*kappa_v^2/Z_base_sys;
    %     Lc(i,1) = Lc_unscale/kappa_p*kappa_v^2/L_base_sys;
    %     rc(i,1) = rc_unscale/kappa_p*kappa_v^2/Z_base_sys;
    %
    %     %
    %     r_outerL(i,1) = rc(i,1);
    %     x_outerL(i,1) = Lc(i,1);
    psi(i,1) = pi/4;
    epsilon(i,1) = 0.1;
    Eb(i,1) = 1*kappa_v/V_base_sys;
    Imax(i,1) = 1.2*(kappa_p/kappa_v)/I_base_sys;
    Li(i,1) = 0.0196/kappa_p*kappa_v^2/L_base_sys;
    Lg(i,1) = 0.0196/kappa_p*kappa_v^2/L_base_sys;
    C(i,1) = 0.1086*kappa_p/kappa_v^2/C_base_sys;
    Ri(i,1) = 0.0139/kappa_p*kappa_v^2/Z_base_sys;
    Rg(i,1) = 0.0139/kappa_p*kappa_v^2/Z_base_sys;
    Kb(i,1) = 0.0347/kappa_p*kappa_v^2/Z_base_sys;
    K_Pi(i,1) = 0.9817/kappa_p*kappa_v^2/Z_base_sys;
    K_Ii(i,1) = 0.6944/kappa_p*kappa_v^2/Z_base_sys;
    k1(i,1) = 0.0033/kappa_p*kappa_v^2/Z_base_sys;
    K_Pv(i,1) = 1.4476/Z_base_sys/kappa_p*kappa_v^2;
    K_Iv(i,1) = 10.2944/Z_base_sys/kappa_p*kappa_v^2;
    wbw_i(i,1) = 50;
    wbw_v(i,1) = 13.33;
    k2(i,1) = 0.0796*1/kappa_v^2/1/V_sys_base^2;
end

Branch_index_outerL = [5;14;20;33;34;37;39;41;46;0];

%% Replace machines with inverters and investigate stability
% inv_vec = (0:9)';
inv_vec = (1:numel(InverterBus))';
nx = 200;
para_fac  = [linspace(0.1,10,nx)];
for ii = 1:length(para_fac)
    Lf = para_fac(ii)*Lf_1;
    rf = para_fac(ii)*rf_1;
    for i_ninv = 1:numel(inv_vec)
        
        N_inv = inv_vec(i_ninv);
        
        % Equilibrium point
        mpc_new.branch = mpc.branch;
        InverterBus_i = [];
        for i = 1:N_inv
            
            InverterBus_i(i) = InverterBus(i);
            
            Branch_index_outerL_i = Branch_index_outerL(i);
            if Branch_index_outerL_i ~= 0
                mpc_new.branch(Branch_index_outerL_i,3:4) = mpc_new.branch(Branch_index_outerL_i,3:4)+[r_outerL(i,1)*0,x_outerL(i,1)*1];
            end
            
        end
        
        results = runpf(mpc_new);
        
        % Kron reduction
        mpc_kron = mpc_new;
        mpc_kron.bus(:,5) = results.bus(:,3)./(results.bus(:,8).^2);
        mpc_kron.bus(:,6) = -results.bus(:,4)./(results.bus(:,8).^2);
        mpc_kron.bus(:,3) = 0;
        mpc_kron.bus(:,4) = 0;
        results_kron = runpf(mpc_kron);
        
        [Ybus, Yf, Yt] = makeYbus(mpc_kron);
        Y_orig = full(Ybus);
        
        Y_aa = Y_orig(1:29,1:29);
        Y_ab = Y_orig(1:29,30:39);
        Y_ba = Y_orig(30:39,1:29);
        Y_bb = Y_orig(30:39,30:39);
        Y_kron = Y_bb - Y_ba*Y_aa^(-1)*Y_ab;
        
        Y_re = real(Y_kron);
        Y_im = imag(Y_kron);
        Y_prime = [Y_re,-Y_im;Y_im,Y_re];
        
        
        for i = 1:10
            
            %         if i <= 10-N_inv
            if ~ismember(i,InverterBus_i)
                %         if i == 1 || i > 1+N_inv
                p_pf(i,1) = results.gen(i,2)*1e6/P_base_sys;
                q_pf(i,1) = results.gen(i,3)*1e6/P_base_sys;
            else
                % disp(i)
                %             p_pf(i,1) = -results.bus(29+i,3)*1e6/P_base_sys;
                %             q_pf(i,1) = -results.bus(29+i,4)*1e6/P_base_sys;
                p_pf(i,1) = results.gen(i,2)*1e6/P_base_sys;
                q_pf(i,1) = results.gen(i,3)*1e6/P_base_sys;
                P_ref_inv(i,1) = p_pf(i,1);
                Q_ref_inv(i,1) = q_pf(i,1);
            end
            
            vm(i,1) = results.bus(29+i,8);
            %         V_ref(i,1) = vm(i,1);
            
            theta(i,1) = results.bus(29+i,9)/180*pi;
            theta_slack = theta - theta(1);
            %     theta_slack = theta;
            
            s(i,1) = complex(p_pf(i,1),q_pf(i,1));
            
        end
        
        
        x_star = [];
        omega = [];
        Eqp = [];
        Edp = [];
        V_f = [];
        V_a = [];
        V_fb = [];
        E_fd = [];
        T_mech = [];
        V_xPSS1 = [];
        V_xPSS2 = [];
        V_xPSS3 = [];
        delta = [];
        i_ld = [];
        i_lq = [];
        v_od = [];
        v_oq = [];
        
        T_mech_eq = [];
        for i = 1:10
            
            if i == 1
                
                It_eq(i,1) = sqrt(p_pf(i,1)^2+q_pf(i,1)^2)/vm(i,1);
                if q_pf(i,1) > 0
                    phi_eq(i,1) = acos(p_pf(i,1)/(vm(i,1)*It_eq(i,1)));
                else
                    phi_eq(i,1) = -acos(p_pf(i,1)/(vm(i,1)*It_eq(i,1)));
                end
                V_phasor(i,1) = complex(vm(i,1),0);
                I_phasor(i,1) = complex(It_eq(i,1)*cos(phi_eq(i,1)),-It_eq(i,1)*sin(phi_eq(i,1)));
                z_phasor(i,1) = complex(Ra(i,1),xq(i,1));
                E_bar(i,1) = V_phasor(i,1) + I_phasor(i,1)*z_phasor(i,1);
                delta_eq(i,1) = angle(E_bar(i,1));
                V_d_eq(i,1) = vm(i,1)*sin(delta_eq(i,1));
                V_q_eq(i,1) = vm(i,1)*cos(delta_eq(i,1));
                I_d_eq(i,1) = It_eq(i,1)*sin(delta_eq(i,1)+phi_eq(i,1));
                I_q_eq(i,1) = It_eq(i,1)*cos(delta_eq(i,1)+phi_eq(i,1));
                phi_d_eq(i,1) = V_q_eq(i,1) + Ra(i,1)*I_q_eq(i,1);
                phi_q_eq(i,1) = -V_d_eq(i,1) - Ra(i,1)*I_d_eq(i,1);
                Edp_eq(i,1) = -phi_q_eq(i,1) - xqp(i,1)*I_q_eq(i,1);
                Eqp_eq(i,1) = phi_d_eq(i,1) + xdp(i,1)*I_d_eq(i,1);
                Efd_eq(i,1) = Eqp_eq(i,1) + (xd(i,1)-xdp(i,1))*I_d_eq(i,1);
                %         V_AVR_eq(i,1) = Efd_eq(i,1)/KA(i,1);
                %         V_e_eq(i,1) = V_AVR_eq(i,1);
                %         V_int_eq(i,1) = vm(i,1) + V_e_eq(i,1);
                %         V_xAVR_eq(i,1) = (TB(i,1)-TC(i,1))*V_AVR_eq(i,1);
                V_fb_eq(i,1) = KF(i,1)/TF(i,1)*Efd_eq(i,1);
                T_e_eq(i,1) = phi_d_eq(i,1)*I_q_eq(i,1) - phi_q_eq(i,1)*I_d_eq(i,1);
                T_mech_eq(i,1) = T_e_eq(i,1);
                V_a_eq(i,1) = Efd_eq(i,1);
                V_ref(i,1) = vm(i,1) + Efd_eq(i,1)/KA(i,1);
                
                x_star_i = [];
                x_star_i(1,1) = 1; % omega
                x_star_i(2,1) = Eqp_eq(i,1); % Eqp
                %         x_star_i(3,1) = Edp_eq(i,1); % Edp
                x_star_i(3,1) = vm(i,1); % V_f
                x_star_i(4,1) = V_a_eq(i,1); % V_a
                x_star_i(5,1) = V_fb_eq(i,1); % V_fb
                x_star_i(6,1) = Efd_eq(i,1); % E_fd
                x_star_i(7,1) = 0; % V_xPSS1
                x_star_i(8,1) = 0; % V_xPSS2
                x_star_i(9,1) = 0; % V_xPSS3
                x_star_i(10,1) = T_mech_eq; % T_mech
                
                x_star = [x_star;x_star_i];
                
                omega(i,1) = x_star_i(1);
                Eqp(i,1) = x_star_i(2);
                %         Edp(i,1) = x_star_i(3);
                V_f(i,1) = x_star_i(3);
                V_a(i,1) = x_star_i(4);
                V_fb(i,1) = x_star_i(5);
                E_fd(i,1) = x_star_i(6);
                V_xPSS1(i,1) = x_star_i(7);
                V_xPSS2(i,1) = x_star_i(8);
                V_xPSS3(i,1) = x_star_i(9);
                T_mech(i,1) = x_star_i(10);
                
                Edp(i,1) = Edp_eq(i,1);
                
                %         elseif i <= 10-N_inv
            elseif ~ismember(i,InverterBus_i)
                
                It_eq(i,1) = sqrt(p_pf(i,1)^2+q_pf(i,1)^2)/vm(i,1);
                if q_pf(i,1) > 0
                    phi_eq(i,1) = acos(p_pf(i,1)/(vm(i,1)*It_eq(i,1)));
                else
                    phi_eq(i,1) = -acos(p_pf(i,1)/(vm(i,1)*It_eq(i,1)));
                end
                V_phasor(i,1) = complex(vm(i,1),0);
                I_phasor(i,1) = complex(It_eq(i,1)*cos(phi_eq(i,1)),-It_eq(i,1)*sin(phi_eq(i,1)));
                z_phasor(i,1) = complex(Ra(i,1),xq(i,1));
                E_bar(i,1) = V_phasor(i,1) + I_phasor(i,1)*z_phasor(i,1);
                delta_eq(i,1) = angle(E_bar(i,1));
                V_d_eq(i,1) = vm(i,1)*sin(delta_eq(i,1));
                V_q_eq(i,1) = vm(i,1)*cos(delta_eq(i,1));
                I_d_eq(i,1) = It_eq(i,1)*sin(delta_eq(i,1)+phi_eq(i,1));
                I_q_eq(i,1) = It_eq(i,1)*cos(delta_eq(i,1)+phi_eq(i,1));
                phi_d_eq(i,1) = V_q_eq(i,1) + Ra(i,1)*I_q_eq(i,1);
                phi_q_eq(i,1) = -V_d_eq(i,1) - Ra(i,1)*I_d_eq(i,1);
                Edp_eq(i,1) = -phi_q_eq(i,1) - xqp(i,1)*I_q_eq(i,1);
                Eqp_eq(i,1) = phi_d_eq(i,1) + xdp(i,1)*I_d_eq(i,1);
                Efd_eq(i,1) = Eqp_eq(i,1) + (xd(i,1)-xdp(i,1))*I_d_eq(i,1);
                %         V_AVR_eq(i,1) = Efd_eq(i,1)/KA(i,1);
                %         V_e_eq(i,1) = V_AVR_eq(i,1);
                %         V_int_eq(i,1) = vm(i,1) + V_e_eq(i,1);
                %         V_xAVR_eq(i,1) = (TB(i,1)-TC(i,1))*V_AVR_eq(i,1);
                V_fb_eq(i,1) = KF(i,1)/TF(i,1)*Efd_eq(i,1);
                T_e_eq(i,1) = phi_d_eq(i,1)*I_q_eq(i,1) - phi_q_eq(i,1)*I_d_eq(i,1);
                T_mech_eq(i,1) = T_e_eq(i,1);
                V_a_eq(i,1) = Efd_eq(i,1);
                V_ref(i,1) = vm(i,1) + Efd_eq(i,1)/KA(i,1);
                
                x_star_i = [];
                x_star_i(1,1) = 1; % omega
                x_star_i(2,1) = Eqp_eq(i,1); % Eqp
                x_star_i(3,1) = Edp_eq(i,1); % Edp
                x_star_i(4,1) = vm(i,1); % V_f
                x_star_i(5,1) = V_a_eq(i,1); % V_a
                x_star_i(6,1) = V_fb_eq(i,1); % V_fb
                x_star_i(7,1) = Efd_eq(i,1); % E_fd
                x_star_i(8,1) = 0; % V_xPSS1
                x_star_i(9,1) = 0; % V_xPSS2
                x_star_i(10,1) = 0; % V_xPSS3
                x_star_i(11,1) = (delta_eq(i,1)+theta_slack(i,1)-delta_eq(1,1))/omega_base_sys; % delta
                x_star_i(12,1) = T_mech_eq(i,1);
                
                x_star = [x_star;x_star_i];
                
                omega(i,1) = x_star_i(1);
                Eqp(i,1) = x_star_i(2);
                Edp(i,1) = x_star_i(3);
                V_f(i,1) = x_star_i(4);
                V_a(i,1) = x_star_i(5);
                V_fb(i,1) = x_star_i(6);
                E_fd(i,1) = x_star_i(7);
                V_xPSS1(i,1) = x_star_i(8);
                V_xPSS2(i,1) = x_star_i(9);
                V_xPSS3(i,1) = x_star_i(10);
                delta(i,1) = x_star_i(11);
                T_mech(i,1) = x_star_i(12);
                
            else
                
                It_eq(i,1) = sqrt(p_pf(i,1)^2+q_pf(i,1)^2)/vm(i,1);
                if q_pf(i,1) > 0
                    phi_eq(i,1) = acos(p_pf(i,1)/(vm(i,1)*It_eq(i,1)));
                else
                    phi_eq(i,1) = -acos(p_pf(i,1)/(vm(i,1)*It_eq(i,1)));
                end
                V_phasor(i,1) = complex(vm(i,1),0);
                I_phasor(i,1) = complex(It_eq(i,1)*cos(phi_eq(i,1)),-It_eq(i,1)*sin(phi_eq(i,1)));
                
                i_od_opf(i,1) = real(I_phasor(i,1));
                i_oq_opf(i,1) = imag(I_phasor(i,1));
                v_od_opf(i,1) = real(V_phasor(i,1));
                v_oq_opf(i,1) = imag(V_phasor(i,1));
                
                A_l = [1/Cf(i,1),-Rd(i,1);Rd(i,1),1/Cf(i,1)];
                B_l = [1/Cf(i,1)*i_od_opf(i,1)-v_oq_opf(i,1)-Rd(i,1)*i_oq_opf(i,1);1/Cf(i,1)*i_oq_opf(i,1)+v_od_opf(i,1)+Rd(i,1)*i_od_opf(i,1)];
                x_l = A_l\B_l;
                i_ld_opf(i,1) = x_l(1);
                i_lq_opf(i,1) = x_l(2);
                
                v_id_opf(i,1) = -Lf(i,1)*i_lq_opf(i,1) + rf(i,1)*i_ld_opf(i,1) + v_od_opf(i,1);
                v_iq_opf(i,1) = Lf(i,1)*i_ld_opf(i,1) + rf(i,1)*i_lq_opf(i,1) + v_oq_opf(i,1);
                
                v_eq(i,1) = sqrt(v_id_opf(i,1)^2+v_iq_opf(i,1)^2);
                theta_L(i,1) = angle(complex(v_id_opf(i,1),v_iq_opf(i,1)));
                delta_eq(i,1) = (theta_L(i,1)+theta_slack(i,1)-delta_eq(1,1)+pi/2);
                
                i_gd_eq(i,1) = cos(-theta_L(i,1))*i_ld_opf(i,1) - sin(-theta_L(i,1))*i_lq_opf(i,1);
                i_gq_eq(i,1) = sin(-theta_L(i,1))*i_ld_opf(i,1) + cos(-theta_L(i,1))*i_lq_opf(i,1);
                v_od_eq(i,1) = cos(-theta_L(i,1))*v_od_opf(i,1) - sin(-theta_L(i,1))*v_oq_opf(i,1);
                v_oq_eq(i,1) = sin(-theta_L(i,1))*v_od_opf(i,1) + cos(-theta_L(i,1))*v_oq_opf(i,1);
                i_od_eq(i,1) = cos(-theta_L(i,1))*v_od_opf(i,1) - sin(-theta_L(i,1))*v_oq_opf(i,1);
                i_oq_eq(i,1) = sin(-theta_L(i,1))*v_od_opf(i,1) + cos(-theta_L(i,1))*v_oq_opf(i,1);
                
                P_st_ref(i,1) = v_id_opf(i,1)*i_ld_opf(i,1) + v_iq_opf(i,1)*i_lq_opf(i,1);
                Q_st_ref(i,1) = v_iq_opf(i,1)*i_ld_opf(i,1) - v_id_opf(i,1)*i_lq_opf(i,1);
                V_nom(i,1) = v_eq(i,1);
                
                x_star_i = [];
                
                
                % 
                
                
                Ud_star
                x_star_i(1,1) = v_eq(i,1); % VOC voltage
                x_star_i(2,1) = delta_eq(i,1)/omega_base_sys; % VOC angle
                x_star_i(3,1) = i_ld_eq(i,1);
                x_star_i(4,1) = i_lq_eq(i,1);
                x_star_i(5,1) = v_od_eq(i,1);
                x_star_i(6,1) = v_oq_eq(i,1);
                x_star_i(7,1) = 1/K_Iv*(Ud_star - v_od_eq(i,1)...
                              - w/wb*Li*T2(pi/2)*i_ld_eq(i,1));
                x_star_i(8,1) = 1/K_Iv*(Uq_star - v_oq_eq(i,1)...
                              - w/wb*Li*T2(pi/2)*i_lq_eq(i,1));
                x_star_i(9,1) = 1/K_Ii*(Ud_star - v_od_eq(i,1)...
                              - w/wb*Li*T2(pi/2)*i_ld_eq(i,1));
                x_star_i(10,1) = 1/K_Ii*(Uq_star - v_oq_eq(i,1)...
                              - w/wb*Li*T2(pi/2)*i_lq_eq(i,1));
                
                x_star = [x_star;x_star_i];
                
                v_iq(i,1)= 0;
                v_id(i,1) = x_star_i(1);
                delta(i,1) = x_star_i(2);
                i_ld(i,1) = x_star_i(3);
                i_lq(i,1) = x_star_i(4);
                v_od(i,1) = x_star_i(5);
                v_oq(i,1) = x_star_i(6);
                
            end
            
        end
        
        % System description
        
        % Network
        C_cur = zeros(20,20);
        C_Edp = zeros(20,10);
        C_Eqp = zeros(20,10);
        v_inv_vec = zeros(20,1);
        phi_vec = zeros(10,1);
        Edp_vec = zeros(10,1);
        Eqp_vec = zeros(10,1);
        f = [];
        ll=[];
        for i = 1:10
            
            if i == 1
                
                kd1(i,1) = -Ra(i,1);
                kd2(i,1) = xqp(i,1);
                kq1(i,1) = -xdp(i,1);
                kq2(i,1) = -Ra(i,1);
                
                C_cur(i,i) = kd1(i,1);
                C_cur(i,i+10) = kd2(i,1);
                C_cur(i+10,i) = kq1(i,1);
                C_cur(i+10,i+10) = kq2(i,1);
                
                C_Edp(i,i) = 1;
                C_Eqp(i+10,i) = 1;
                
                Edp_vec(i,1) = Edp(i,1);
                Eqp_vec(i,1) = Eqp(i,1);
                ll (i) = Edp(i,1);
                
            elseif ~ismember(i,InverterBus_i)
                %         elseif i <= 10-N_inv
                %        elseif i > 1+N_inv
                
                kd1(i,1) = -Ra(i,1);
                kd2(i,1) = xqp(i,1);
                kq1(i,1) = -xdp(i,1);
                kq2(i,1) = -Ra(i,1);
                
                k_mat = [kd1(i,1),kd2(i,1);kq1(i,1),kq2(i,1)];
                phi_mat = [cos(delta(i,1)*omega_base_sys),-sin(delta(i,1)*omega_base_sys);sin(delta(i,1)*omega_base_sys),cos(delta(i,1)*omega_base_sys)];
                phi_mat_inv = [cos(-delta(i,1)*omega_base_sys),-sin(-delta(i,1)*omega_base_sys);sin(-delta(i,1)*omega_base_sys),cos(-delta(i,1)*omega_base_sys)];
                k_mat_rot = phi_mat*k_mat*phi_mat_inv;
                kd1_rot(i,1) = k_mat_rot(1,1);
                kd2_rot(i,1) = k_mat_rot(1,2);
                kq1_rot(i,1) = k_mat_rot(2,1);
                kq2_rot(i,1) = k_mat_rot(2,2);
                
                C_cur(i,i) = kd1_rot(i,1);
                C_cur(i,i+10) = kd2_rot(i,1);
                C_cur(i+10,i) = kq1_rot(i,1);
                C_cur(i+10,i+10) = kq2_rot(i,1);
                
                C_Edp(i,i) = phi_mat(1,1);
                C_Eqp(i,i) = phi_mat(1,2);
                
                C_Edp(i+10,i) = phi_mat(2,1);
                C_Eqp(i+10,i) = phi_mat(2,2);
                
                Edp_vec(i,1) = Edp(i,1);
                Eqp_vec(i,1) = Eqp(i,1);
                
            else
                
                %             v_od_opf(i,1) = vm(i,1);
                %             v_oq_opf(i,1) = 0;
                %             delta_opf(i,1) = (theta_slack(i,1)-delta_eq(1,1)+pi/2)/omega_base_sys;
                
                v_oD(i,1) = cos(delta(i,1)*omega_base_sys)*v_od(i,1) - sin(delta(i,1)*omega_base_sys)*v_oq(i,1);
                v_oQ(i,1) = sin(delta(i,1)*omega_base_sys)*v_od(i,1) + cos(delta(i,1)*omega_base_sys)*v_oq(i,1);
                
                v_inv_vec(i) = v_oD(i,1);
                v_inv_vec(i+10) = v_oQ(i,1);
                
            end
            
            
        end
        
        i_vec = (eye(20)-Y_prime*C_cur)\(Y_prime*v_inv_vec+Y_prime*(C_Edp*Edp_vec+C_Eqp*Eqp_vec));
        % i_vec = Y_prime*v_inv_vec;
        
        
        % System dynamics
        % x = [omega, Eqp, Edp, V_f, V_int, V_xAVR, E_fd, V_xPSS1, V_xPSS2, V_xPSS3, delta]
        for i = 1:10
            
            if i == 1
                
                i_mD(i,1) = i_vec(i);
                i_mQ(i,1) = i_vec(i+10);
                
                i_md(i,1) = i_mD(i,1);
                i_mq(i,1) = i_mQ(i,1);
                
                V_PSS1(i,1) = (omega(i,1)-omega_nom)*Kpss(i,1) - V_xPSS1(i,1);
                V_PSS2(i,1) = 1/T2(i,1)*(T1(i,1)*V_PSS1(i,1) + V_xPSS2(i,1));
                V_PSS(i,1) = 1/T4(i,1)*(T3(i,1)*V_PSS2(i,1) + V_xPSS3(i,1));
                phi_d(i,1) = -xdp(i,1)*i_md(i,1) + Eqp(i,1);
                phi_q(i,1) = -xqp(i,1)*i_mq(i,1) - Edp(i,1);
                V_d(i,1) = -Ra(i,1)*i_md(i,1) - phi_q(i,1);
                V_q(i,1) = -Ra(i,1)*i_mq(i,1) + phi_d(i,1);
                V_T(i,1) = sqrt(V_d(i,1)^2+V_q(i,1)^2);
                V_in(i,1) = V_ref(i,1) - V_T(i,1) + V_fb(i,1) - KF(i,1)/TF(i,1)*E_fd(i,1) + V_PSS(i,1);
                T_e(i,1) = phi_d(i,1)*i_mq(i,1) - phi_q(i,1)*i_md(i,1);
                %             T_mech(i,1) = T_mech_eq(i,1);
                
                
                
                % x = [Eqp, Edp, delta, omega, V_f, V_a, V_fb, E_fd, V_xPSS1, V_xPSS2, V_xPSS3]
                f_i = [];
                
                f_i(1,1) = 1/M(i,1) * (T_mech(i,1) - T_e(i,1) - D(i,1)*(omega(i,1)-omega_nom));
                f_i(2,1) = 1/Td0p(i,1) * (-Eqp(i,1) - (xd(i,1)-xdp(i,1))*i_md(i,1) + E_fd(i,1));
                %         f_i(3,1) = 1/Tq0p(i,1) * (-Edp(i,1) + (xq(i,1)-xqp(i,1))*i_mq(i,1));
                f_i(3,1) = 1/TR(i,1) * (-V_f(i,1) + V_T(i,1));
                f_i(4,1) = 1/TA(i,1)*(-V_a(i,1) + KA(i,1)*V_in(i,1));
                f_i(5,1) = 1/TF(i,1)*(-V_fb(i,1) + KF(i,1)/TF(i,1)*E_fd(i,1));
                f_i(6,1) = 1/TE(i,1) * (-E_fd(i,1) + V_a(i,1));
                f_i(7,1) = V_PSS1(i,1)/TW(i,1);
                f_i(8,1) = -V_PSS2(i,1) + V_PSS1(i,1);
                f_i(9,1) = -V_PSS(i,1) + V_PSS2(i,1);
                f_i(10,1) = 1/TM(i,1)*(-(T_mech(i,1)-T_mech_eq(i,1)) - 1/(RD(i,1)/sysdata_gen_Pmax(i,1)*100)*(omega(i,1)-omega_nom)); % T_mech
                
                f = [f;f_i];
                
                
            elseif ~ismember(i,InverterBus_i)
                %         elseif i <= 10-N_inv
                %         elseif i > 1+N_inv
                
                i_mD(i,1) = i_vec(i);
                i_mQ(i,1) = i_vec(i+10);
                
                i_md(i,1) = cos(-delta(i,1)*omega_base_sys)*i_mD(i,1) - sin(-delta(i,1)*omega_base_sys)*i_mQ(i,1);
                i_mq(i,1) = sin(-delta(i,1)*omega_base_sys)*i_mD(i,1) + cos(-delta(i,1)*omega_base_sys)*i_mQ(i,1);
                
                %         i_md(i,1) = I_d_eq(i,1);
                %         i_mq(i,1) = I_q_eq(i,1);
                
                
                V_PSS1(i,1) = (omega(i,1)-omega_nom)*Kpss(i,1) - V_xPSS1(i,1);
                V_PSS2(i,1) = 1/T2(i,1)*(T1(i,1)*V_PSS1(i,1) + V_xPSS2(i,1));
                V_PSS(i,1) = 1/T4(i,1)*(T3(i,1)*V_PSS2(i,1) + V_xPSS3(i,1));
                phi_d(i,1) = -xdp(i,1)*i_md(i,1) + Eqp(i,1);
                phi_q(i,1) = -xqp(i,1)*i_mq(i,1) - Edp(i,1);
                V_d(i,1) = -Ra(i,1)*i_md(i,1) - phi_q(i,1);
                V_q(i,1) = -Ra(i,1)*i_mq(i,1) + phi_d(i,1);
                V_T(i,1) = sqrt(V_d(i,1)^2+V_q(i,1)^2);
                V_in(i,1) = V_ref(i,1) - V_T(i,1) + V_fb(i,1) - KF(i,1)/TF(i,1)*E_fd(i,1) + V_PSS(i,1);
                T_e(i,1) = phi_d(i,1)*i_mq(i,1) - phi_q(i,1)*i_md(i,1);
                %             T_mech(i,1) = T_mech_eq(i,1);
                
                f_i = [];
                
                f_i(1,1) = 1/M(i,1) * (T_mech(i,1) - T_e(i,1) - D(i,1)*(omega(i,1)-omega_nom));
                f_i(2,1) = 1/Td0p(i,1) * (-Eqp(i,1) - (xd(i,1)-xdp(i,1))*i_md(i,1) + E_fd(i,1));
                f_i(3,1) = 1/Tq0p(i,1) * (-Edp(i,1) + (xq(i,1)-xqp(i,1))*i_mq(i,1));
                f_i(4,1) = 1/TR(i,1) * (-V_f(i,1) + V_T(i,1));
                f_i(5,1) = 1/TA(i,1)*(-V_a(i,1) + KA(i,1)*V_in(i,1));
                f_i(6,1) = 1/TF(i,1)*(-V_fb(i,1) + KF(i,1)/TF(i,1)*E_fd(i,1));
                f_i(7,1) = 1/TE(i,1) * (-E_fd(i,1) + V_a(i,1));
                f_i(8,1) = V_PSS1(i,1)/TW(i,1);
                f_i(9,1) = -V_PSS2(i,1) + V_PSS1(i,1);
                f_i(10,1) = -V_PSS(i,1) + V_PSS2(i,1);
                f_i(11,1) = omega(i,1) - omega(1,1);
                f_i(12,1) = 1/TM(i,1)*(-(T_mech(i,1)-T_mech_eq(i,1)) - 1/(RD(i,1)/sysdata_gen_Pmax(i,1)*100)*(omega(i,1)-omega_nom)); % T_mech
                
                f = [f;f_i];
                
            else
                
                i_oD(i,1) = i_vec(i);
                i_oQ(i,1) = i_vec(i+10);
                
                i_od(i,1) = cos(-delta(i,1)*omega_base_sys)*i_oD(i,1) - sin(-delta(i,1)*omega_base_sys)*i_oQ(i,1);
                i_oq(i,1) = sin(-delta(i,1)*omega_base_sys)*i_oD(i,1) + cos(-delta(i,1)*omega_base_sys)*i_oQ(i,1);
                
                %             f_i = [];
                %
                %             f_i(1,1) = xi(i,1)/(mu_v(i,1)^2)*v_id(i,1)*2*(V_nom(i,1)^2 - v_id(i,1)^2) - mu_v(i,1)*mu_i(i,1)/(3*Cvoc(i,1)*v_id(i,1))*(-v_id(i,1)*i_lq(i,1) - Q_st_ref(i,1)); % VOC voltage
                %             f_i(2,1) = omega_nom - (mu_v(i,1)*mu_i(i,1)/(3*Cvoc(i,1)*v_id(i,1)^2)*(v_id(i,1)*i_ld(i,1) - P_st_ref(i,1)))/omega_base_sys - omega(1,1);
                %             f_i(3,1) = (1/Lf(i,1)*(v_id(i,1) - v_od(i,1) - rf(i,1)*i_ld(i,1)) + omega_nom*i_lq(i,1))*omega_base_sys; %ild
                %             f_i(4,1) = (1/Lf(i,1)*(v_iq(i,1) - v_oq(i,1) - rf(i,1)*i_lq(i,1)) - omega_nom*i_ld(i,1))*omega_base_sys; %ilq
                %             f_i(5,1) = Rd(i,1)*f_i(3,1) + (1/Cf(i,1)*(i_ld(i,1) - i_od(i,1)) + omega_nom*v_oq(i,1) - omega_nom*Rd(i,1)*(i_lq(i,1) - i_oq(i,1)))*omega_base_sys; %vod
                %             f_i(6,1) = Rd(i,1)*f_i(4,1) + (1/Cf(i,1)*(i_lq(i,1) - i_oq(i,1)) - omega_nom*v_od(i,1) + omega_nom*Rd(i,1)*(i_ld(i,1) - i_od(i,1)))*omega_base_sys; %voq
                %
                
                e1 = [1 0]'; e2 = [0 1]';
                S_star = [P_star;Q_star];
                 % P and Q:
                P = Edq'*Igdq;
                Q = Edq'*T2(-pi/2)*Igdq;
                
                S = [P;Q]; 
                % theta
                f_i(1,1) = wb(i,1) + wb(i,1)*k1(i,1)/E_star^2* e1'*T2(psi(i,1) - pi/2) * (S_star - S);
                
                
                % E_star
                f_i(2,1) = wb(i,1)*k1(i,1)/E_star^2 * e2'*T2(psi(i,1) - pi/2)...
                    * (S_star - S)... % E_star
                    + wb(i,1) *k2(i,1)*(Eb^2 - E_star^2)*E_star;
                
                % Iidq
                f_i(3:4,1) = (w*T2(pi/2) - wb*Ri(i,1)/Li(i,1)*eye(2)) * Iidq...
                    + wb(i,1)/Li(i,1)*(Udq - Edq);
                
                % Edq
                f_i(5:6,1) = w*T2(pi/2)*Edq + wb(i,1)/C(i,1)*(Iidq - Igdq);
                
                % add later if needed
%                 % Igdq
%                 f_i(7:8,1) = (w*T2(pi/2) - wb*Rg/Lg*eye(2))*Igdq...
%                     + wb/Lg*(Edq - T2(delta)*VDQ);
                
               
                
                
                rho = 1.2/norm(Iidq)^2;
                
                % Voltage controller:
                f_i(7:8) = wb(i,1)*(e1*E_star - Edq) + wb(i,1)*Kb(i,1)*(rho(i,1)-1)...
                    *Iidq_star;
                rho_cap = 1 + K_Pv(i,1) * Kb(i,1) * (rho - 1);
                Iidq_star = 1/rho_cap*(K_Pv(i,1)/wb(i,1)*f_i(9:10)...
                    + K_Iv(i,1)*phi_dq + Igdq -...
                    w/wb(i,1)*C(i,1)*T2(pi/2)*Edq);
                
                
                % Current controller:
                f_i(9:10) = wb*(rho*Iidq_star - Iidq);
                Udq_star = K_Pi/wb * wb*(rho*Iidq_star - Iidq) + ...
                            K_Ii*Gamma_dq + Edq - w/wb*Li*T2(pi/2)*Iidq;
                
                f = [f;f_i];
                
            end
            
        end
        
        
        f_eq = f;
        f_eq_mat{i_ninv,1} = f_eq;
        err_eq = norm(f);
        err_eq_vec(i_ninv,1) = err_eq;
        
        
        % Linearize system, Jacobian
        
        x_star_ind = zeros(10,2);
        for i_xstar = 1:10
            if i_xstar == 1
                x_star_ind(i_xstar,1) = 1;
            else
                x_star_ind(i_xstar,1) = x_star_ind(i_xstar-1,2)+1;
            end
            if i_xstar == 1
                x_star_ind(i_xstar,2) = x_star_ind(i_xstar,1)+10-1;
            elseif ~ismember(i_xstar,InverterBus_i)
                x_star_ind(i_xstar,2) = x_star_ind(i_xstar,1)+12-1;
            else
                x_star_ind(i_xstar,2) = x_star_ind(i_xstar,1)+6-1;
            end
        end
        
        dx = 1e-6;
        dfdx = [];
        for j = 1:numel(x_star)
            
            dx_plus = zeros(numel(x_star),1);
            dx_plus(j,1) = dx;
            x_star_j = x_star + dx_plus;
            
            for i = 1:10
                
                if i == 1
                    
                    %                 x_star_i = x_star_j(1:9);
                    x_star_i = x_star_j(x_star_ind(i,1):x_star_ind(i,2));
                    
                    omega(i,1) = x_star_i(1);
                    Eqp(i,1) = x_star_i(2);
                    %             Edp(i,1) = x_star_i(3);
                    V_f(i,1) = x_star_i(3);
                    V_a(i,1) = x_star_i(4);
                    V_fb(i,1) = x_star_i(5);
                    E_fd(i,1) = x_star_i(6);
                    V_xPSS1(i,1) = x_star_i(7);
                    V_xPSS2(i,1) = x_star_i(8);
                    V_xPSS3(i,1) = x_star_i(9);
                    T_mech(i,1) = x_star_i(10);
                    
                    Edp(i,1) = Edp_eq(i,1);
                    
                    %             elseif i <= 10-N_inv
                elseif ~ismember(i,InverterBus_i)
                    
                    %                 x_star_i = x_star_j((i-2)*11+10:(i-1)*11+9);
                    x_star_i = x_star_j(x_star_ind(i,1):x_star_ind(i,2));
                    
                    omega(i,1) = x_star_i(1);
                    Eqp(i,1) = x_star_i(2);
                    Edp(i,1) = x_star_i(3);
                    V_f(i,1) = x_star_i(4);
                    V_a(i,1) = x_star_i(5);
                    V_fb(i,1) = x_star_i(6);
                    E_fd(i,1) = x_star_i(7);
                    V_xPSS1(i,1) = x_star_i(8);
                    V_xPSS2(i,1) = x_star_i(9);
                    V_xPSS3(i,1) = x_star_i(10);
                    delta(i,1) = x_star_i(11);
                    T_mech(i,1) = x_star_i(12);
                    
                else
                    
                    %                 x_star_i = x_star_j((i-(10-N_inv)-1)*6+9+(10-N_inv-1)*11+1:(i-(10-N_inv))*6+9+(10-N_inv-1)*11);
                    x_star_i = x_star_j(x_star_ind(i,1):x_star_ind(i,2));
                    
                    v_iq(i,1)= 0;
                    v_id(i,1) = x_star_i(1);
                    delta(i,1) = x_star_i(2);
                    i_ld(i,1) = x_star_i(3);
                    i_lq(i,1) = x_star_i(4);
                    v_od(i,1) = x_star_i(5);
                    v_oq(i,1) = x_star_i(6);
                    
                end
                %
            end
            
            C_cur = zeros(20,20);
            C_Edp = zeros(20,10);
            C_Eqp = zeros(20,10);
            v_inv_vec = zeros(20,1);
            phi_vec = zeros(10,1);
            f = [];
            for i = 1:10
                
                if i == 1
                    
                    kd1(i,1) = -Ra(i,1);
                    kd2(i,1) = xqp(i,1);
                    kq1(i,1) = -xdp(i,1);
                    kq2(i,1) = -Ra(i,1);
                    
                    C_cur(i,i) = kd1(i,1);
                    C_cur(i,i+10) = kd2(i,1);
                    C_cur(i+10,i) = kq1(i,1);
                    C_cur(i+10,i+10) = kq2(i,1);
                    
                    C_Edp(i,i) = 1;
                    C_Eqp(i+10,i) = 1;
                    
                    Edp_vec(i,1) = Edp(i,1);
                    Eqp_vec(i,1) = Eqp(i,1);
                    
                    %             elseif i <= 10-N_inv
                elseif ~ismember(i,InverterBus_i)
                    
                    kd1(i,1) = -Ra(i,1);
                    kd2(i,1) = xqp(i,1);
                    kq1(i,1) = -xdp(i,1);
                    kq2(i,1) = -Ra(i,1);
                    
                    k_mat = [kd1(i,1),kd2(i,1);kq1(i,1),kq2(i,1)];
                    phi_mat = [cos(delta(i,1)*omega_base_sys),-sin(delta(i,1)*omega_base_sys);sin(delta(i,1)*omega_base_sys),cos(delta(i,1)*omega_base_sys)];
                    phi_mat_inv = [cos(-delta(i,1)*omega_base_sys),-sin(-delta(i,1)*omega_base_sys);sin(-delta(i,1)*omega_base_sys),cos(-delta(i,1)*omega_base_sys)];
                    k_mat_rot = phi_mat*k_mat*phi_mat_inv;
                    kd1_rot(i,1) = k_mat_rot(1,1);
                    kd2_rot(i,1) = k_mat_rot(1,2);
                    kq1_rot(i,1) = k_mat_rot(2,1);
                    kq2_rot(i,1) = k_mat_rot(2,2);
                    
                    C_cur(i,i) = kd1_rot(i,1);
                    C_cur(i,i+10) = kd2_rot(i,1);
                    C_cur(i+10,i) = kq1_rot(i,1);
                    C_cur(i+10,i+10) = kq2_rot(i,1);
                    
                    C_Edp(i,i) = phi_mat(1,1);
                    C_Eqp(i,i) = phi_mat(1,2);
                    
                    C_Edp(i+10,i) = phi_mat(2,1);
                    C_Eqp(i+10,i) = phi_mat(2,2);
                    
                    Edp_vec(i,1) = Edp(i,1);
                    Eqp_vec(i,1) = Eqp(i,1);
                    
                else
                    
                    v_oD(i,1) = cos(delta(i,1)*omega_base_sys)*v_od(i,1) - sin(delta(i,1)*omega_base_sys)*v_oq(i,1);
                    v_oQ(i,1) = sin(delta(i,1)*omega_base_sys)*v_od(i,1) + cos(delta(i,1)*omega_base_sys)*v_oq(i,1);
                    
                    v_inv_vec(i) = v_oD(i,1);
                    v_inv_vec(i+10) = v_oQ(i,1);
                    
                end
            end
            
            i_vec = (eye(20)-Y_prime*C_cur)\(Y_prime*v_inv_vec+Y_prime*(C_Edp*Edp_vec+C_Eqp*Eqp_vec));
            
            for i = 1:10
                
                if i == 1
                    
                    i_mD(i,1) = i_vec(i);
                    i_mQ(i,1) = i_vec(i+10);
                    
                    i_md(i,1) = i_mD(i,1);
                    i_mq(i,1) = i_mQ(i,1);
                    
                    V_PSS1(i,1) = (omega(i,1)-omega_nom)*Kpss(i,1) - V_xPSS1(i,1);
                    V_PSS2(i,1) = 1/T2(i,1)*(T1(i,1)*V_PSS1(i,1) + V_xPSS2(i,1));
                    V_PSS(i,1) = 1/T4(i,1)*(T3(i,1)*V_PSS2(i,1) + V_xPSS3(i,1));
                    phi_d(i,1) = -xdp(i,1)*i_md(i,1) + Eqp(i,1);
                    phi_q(i,1) = -xqp(i,1)*i_mq(i,1) - Edp(i,1);
                    V_d(i,1) = -Ra(i,1)*i_md(i,1) - phi_q(i,1);
                    V_q(i,1) = -Ra(i,1)*i_mq(i,1) + phi_d(i,1);
                    V_T(i,1) = sqrt(V_d(i,1)^2+V_q(i,1)^2);
                    V_in(i,1) = V_ref(i,1) - V_T(i,1) + V_fb(i,1) - KF(i,1)/TF(i,1)*E_fd(i,1) + V_PSS(i,1);
                    T_e(i,1) = phi_d(i,1)*i_mq(i,1) - phi_q(i,1)*i_md(i,1);
                    %                 T_mech(i,1) = T_mech_eq(i,1);
                    
                    
                    
                    % x = [Eqp, Edp, delta, omega, V_f, V_a, V_fb, E_fd, V_xPSS1, V_xPSS2, V_xPSS3]
                    f_i = [];
                    
                    f_i(1,1) = 1/M(i,1) * (T_mech(i,1) - T_e(i,1) - D(i,1)*(omega(i,1)-omega_nom));
                    f_i(2,1) = 1/Td0p(i,1) * (-Eqp(i,1) - (xd(i,1)-xdp(i,1))*i_md(i,1) + E_fd(i,1));
                    %         f_i(3,1) = 1/Tq0p(i,1) * (-Edp(i,1) + (xq(i,1)-xqp(i,1))*i_mq(i,1));
                    f_i(3,1) = 1/TR(i,1) * (-V_f(i,1) + V_T(i,1));
                    f_i(4,1) = 1/TA(i,1)*(-V_a(i,1) + KA(i,1)*V_in(i,1));
                    f_i(5,1) = 1/TF(i,1)*(-V_fb(i,1) + KF(i,1)/TF(i,1)*E_fd(i,1));
                    f_i(6,1) = 1/TE(i,1) * (-E_fd(i,1) + V_a(i,1));
                    f_i(7,1) = V_PSS1(i,1)/TW(i,1);
                    f_i(8,1) = -V_PSS2(i,1) + V_PSS1(i,1);
                    f_i(9,1) = -V_PSS(i,1) + V_PSS2(i,1);
                    f_i(10,1) = 1/TM(i,1)*(-(T_mech(i,1)-T_mech_eq(i,1)) - 1/(RD(i,1)/sysdata_gen_Pmax(i,1)*100)*(omega(i,1)-omega_nom)); % T_mech
                    
                    f = [f;f_i];
                    
                    %             elseif i <= 10-N_inv
                elseif ~ismember(i,InverterBus_i)
                    
                    i_mD(i,1) = i_vec(i);
                    i_mQ(i,1) = i_vec(i+10);
                    
                    i_md(i,1) = cos(-delta(i,1)*omega_base_sys)*i_mD(i,1) - sin(-delta(i,1)*omega_base_sys)*i_mQ(i,1);
                    i_mq(i,1) = sin(-delta(i,1)*omega_base_sys)*i_mD(i,1) + cos(-delta(i,1)*omega_base_sys)*i_mQ(i,1);
                    
                    %         i_md(i,1) = I_d_eq(i,1);
                    %         i_mq(i,1) = I_q_eq(i,1);
                    
                    
                    V_PSS1(i,1) = (omega(i,1)-omega_nom)*Kpss(i,1) - V_xPSS1(i,1);
                    V_PSS2(i,1) = 1/T2(i,1)*(T1(i,1)*V_PSS1(i,1) + V_xPSS2(i,1));
                    V_PSS(i,1) = 1/T4(i,1)*(T3(i,1)*V_PSS2(i,1) + V_xPSS3(i,1));
                    phi_d(i,1) = -xdp(i,1)*i_md(i,1) + Eqp(i,1);
                    phi_q(i,1) = -xqp(i,1)*i_mq(i,1) - Edp(i,1);
                    V_d(i,1) = -Ra(i,1)*i_md(i,1) - phi_q(i,1);
                    V_q(i,1) = -Ra(i,1)*i_mq(i,1) + phi_d(i,1);
                    V_T(i,1) = sqrt(V_d(i,1)^2+V_q(i,1)^2);
                    V_in(i,1) = V_ref(i,1) - V_T(i,1) + V_fb(i,1) - KF(i,1)/TF(i,1)*E_fd(i,1) + V_PSS(i,1);
                    T_e(i,1) = phi_d(i,1)*i_mq(i,1) - phi_q(i,1)*i_md(i,1);
                    %                 T_mech(i,1) = T_mech_eq(i,1);
                    
                    f_i = [];
                    
                    f_i(1,1) = 1/M(i,1) * (T_mech(i,1) - T_e(i,1) - D(i,1)*(omega(i,1)-omega_nom));
                    f_i(2,1) = 1/Td0p(i,1) * (-Eqp(i,1) - (xd(i,1)-xdp(i,1))*i_md(i,1) + E_fd(i,1));
                    f_i(3,1) = 1/Tq0p(i,1) * (-Edp(i,1) + (xq(i,1)-xqp(i,1))*i_mq(i,1));
                    f_i(4,1) = 1/TR(i,1) * (-V_f(i,1) + V_T(i,1));
                    f_i(5,1) = 1/TA(i,1)*(-V_a(i,1) + KA(i,1)*V_in(i,1));
                    f_i(6,1) = 1/TF(i,1)*(-V_fb(i,1) + KF(i,1)/TF(i,1)*E_fd(i,1));
                    f_i(7,1) = 1/TE(i,1) * (-E_fd(i,1) + V_a(i,1));
                    f_i(8,1) = V_PSS1(i,1)/TW(i,1);
                    f_i(9,1) = -V_PSS2(i,1) + V_PSS1(i,1);
                    f_i(10,1) = -V_PSS(i,1) + V_PSS2(i,1);
                    f_i(11,1) = omega(i,1) - omega(1,1);
                    f_i(12,1) = 1/TM(i,1)*(-(T_mech(i,1)-T_mech_eq(i,1)) - 1/(RD(i,1)/sysdata_gen_Pmax(i,1)*100)*(omega(i,1)-omega_nom)); % T_mech
                    
                    f = [f;f_i];
                    
                else
                    
                    i_oD(i,1) = i_vec(i);
                    i_oQ(i,1) = i_vec(i+10);
                    
                    i_od(i,1) = cos(-delta(i,1)*omega_base_sys)*i_oD(i,1) - sin(-delta(i,1)*omega_base_sys)*i_oQ(i,1);
                    i_oq(i,1) = sin(-delta(i,1)*omega_base_sys)*i_oD(i,1) + cos(-delta(i,1)*omega_base_sys)*i_oQ(i,1);
                    
                    f_i = [];
                    
                    f_i(1,1) = xi(i,1)/(mu_v(i,1)^2)*v_id(i,1)*2*(V_nom(i,1)^2 - v_id(i,1)^2) - mu_v(i,1)*mu_i(i,1)/(3*Cvoc(i,1)*v_id(i,1))*(-v_id(i,1)*i_lq(i,1) - Q_st_ref(i,1)); % VOC voltage
                    f_i(2,1) = omega_nom - (mu_v(i,1)*mu_i(i,1)/(3*Cvoc(i,1)*v_id(i,1)^2)*(v_id(i,1)*i_ld(i,1) - P_st_ref(i,1)))/omega_base_sys - omega(1,1);
                    f_i(3,1) = (1/Lf(i,1)*(v_id(i,1) - v_od(i,1) - rf(i,1)*i_ld(i,1)) + omega_nom*i_lq(i,1))*omega_base_sys; %ild
                    f_i(4,1) = (1/Lf(i,1)*(v_iq(i,1) - v_oq(i,1) - rf(i,1)*i_lq(i,1)) - omega_nom*i_ld(i,1))*omega_base_sys; %ilq
                    f_i(5,1) = Rd(i,1)*f_i(3,1) + (1/Cf(i,1)*(i_ld(i,1) - i_od(i,1)) + omega_nom*v_oq(i,1) - omega_nom*Rd(i,1)*(i_lq(i,1) - i_oq(i,1)))*omega_base_sys; %vod
                    f_i(6,1) = Rd(i,1)*f_i(4,1) + (1/Cf(i,1)*(i_lq(i,1) - i_oq(i,1)) - omega_nom*v_od(i,1) + omega_nom*Rd(i,1)*(i_ld(i,1) - i_od(i,1)))*omega_base_sys; %voq
                    
                    f = [f;f_i];
                    
                end
                
            end
            
            f_plus = f;
            
            
            dx_minus = zeros(numel(x_star),1);
            dx_minus(j,1) = -dx;
            x_star_j = x_star + dx_minus;
            
            for i = 1:10
                
                if i == 1
                    
                    %                 x_star_i = x_star_j(1:9);
                    x_star_i = x_star_j(x_star_ind(i,1):x_star_ind(i,2));
                    
                    omega(i,1) = x_star_i(1);
                    Eqp(i,1) = x_star_i(2);
                    %             Edp(i,1) = x_star_i(3);
                    V_f(i,1) = x_star_i(3);
                    V_a(i,1) = x_star_i(4);
                    V_fb(i,1) = x_star_i(5);
                    E_fd(i,1) = x_star_i(6);
                    V_xPSS1(i,1) = x_star_i(7);
                    V_xPSS2(i,1) = x_star_i(8);
                    V_xPSS3(i,1) = x_star_i(9);
                    T_mech(i,1) = x_star_i(10);
                    
                    Edp(i,1) = Edp_eq(i,1);
                    
                    %             elseif i <= 10-N_inv
                elseif ~ismember(i,InverterBus_i)
                    
                    %                 x_star_i = x_star_j((i-2)*11+10:(i-1)*11+9);
                    x_star_i = x_star_j(x_star_ind(i,1):x_star_ind(i,2));
                    
                    omega(i,1) = x_star_i(1);
                    Eqp(i,1) = x_star_i(2);
                    Edp(i,1) = x_star_i(3);
                    V_f(i,1) = x_star_i(4);
                    V_a(i,1) = x_star_i(5);
                    V_fb(i,1) = x_star_i(6);
                    E_fd(i,1) = x_star_i(7);
                    V_xPSS1(i,1) = x_star_i(8);
                    V_xPSS2(i,1) = x_star_i(9);
                    V_xPSS3(i,1) = x_star_i(10);
                    delta(i,1) = x_star_i(11);
                    T_mech(i,1) = x_star_i(12);
                    
                else
                    
                    %                 x_star_i = x_star_j((i-(10-N_inv)-1)*6+9+(10-N_inv-1)*11+1:(i-(10-N_inv))*6+9+(10-N_inv-1)*11);
                    x_star_i = x_star_j(x_star_ind(i,1):x_star_ind(i,2));
                    
                    v_iq(i,1)= 0;
                    v_id(i,1) = x_star_i(1);
                    delta(i,1) = x_star_i(2);
                    i_ld(i,1) = x_star_i(3);
                    i_lq(i,1) = x_star_i(4);
                    v_od(i,1) = x_star_i(5);
                    v_oq(i,1) = x_star_i(6);
                    
                end
                %
            end
            
            C_cur = zeros(20,20);
            C_Edp = zeros(20,10);
            C_Eqp = zeros(20,10);
            v_inv_vec = zeros(20,1);
            phi_vec = zeros(10,1);
            f = [];
            for i = 1:10
                
                if i == 1
                    
                    kd1(i,1) = -Ra(i,1);
                    kd2(i,1) = xqp(i,1);
                    kq1(i,1) = -xdp(i,1);
                    kq2(i,1) = -Ra(i,1);
                    
                    C_cur(i,i) = kd1(i,1);
                    C_cur(i,i+10) = kd2(i,1);
                    C_cur(i+10,i) = kq1(i,1);
                    C_cur(i+10,i+10) = kq2(i,1);
                    
                    C_Edp(i,i) = 1;
                    C_Eqp(i+10,i) = 1;
                    
                    Edp_vec(i,1) = Edp(i,1);
                    Eqp_vec(i,1) = Eqp(i,1);
                    
                    %             elseif i <= 10-N_inv
                elseif ~ismember(i,InverterBus_i)
                    
                    kd1(i,1) = -Ra(i,1);
                    kd2(i,1) = xqp(i,1);
                    kq1(i,1) = -xdp(i,1);
                    kq2(i,1) = -Ra(i,1);
                    
                    k_mat = [kd1(i,1),kd2(i,1);kq1(i,1),kq2(i,1)];
                    phi_mat = [cos(delta(i,1)*omega_base_sys),-sin(delta(i,1)*omega_base_sys);sin(delta(i,1)*omega_base_sys),cos(delta(i,1)*omega_base_sys)];
                    phi_mat_inv = [cos(-delta(i,1)*omega_base_sys),-sin(-delta(i,1)*omega_base_sys);sin(-delta(i,1)*omega_base_sys),cos(-delta(i,1)*omega_base_sys)];
                    k_mat_rot = phi_mat*k_mat*phi_mat_inv;
                    kd1_rot(i,1) = k_mat_rot(1,1);
                    kd2_rot(i,1) = k_mat_rot(1,2);
                    kq1_rot(i,1) = k_mat_rot(2,1);
                    kq2_rot(i,1) = k_mat_rot(2,2);
                    
                    C_cur(i,i) = kd1_rot(i,1);
                    C_cur(i,i+10) = kd2_rot(i,1);
                    C_cur(i+10,i) = kq1_rot(i,1);
                    C_cur(i+10,i+10) = kq2_rot(i,1);
                    
                    C_Edp(i,i) = phi_mat(1,1);
                    C_Eqp(i,i) = phi_mat(1,2);
                    
                    C_Edp(i+10,i) = phi_mat(2,1);
                    C_Eqp(i+10,i) = phi_mat(2,2);
                    
                    Edp_vec(i,1) = Edp(i,1);
                    Eqp_vec(i,1) = Eqp(i,1);
                    
                else
                    
                    v_oD(i,1) = cos(delta(i,1)*omega_base_sys)*v_od(i,1) - sin(delta(i,1)*omega_base_sys)*v_oq(i,1);
                    v_oQ(i,1) = sin(delta(i,1)*omega_base_sys)*v_od(i,1) + cos(delta(i,1)*omega_base_sys)*v_oq(i,1);
                    
                    v_inv_vec(i) = v_oD(i,1);
                    v_inv_vec(i+10) = v_oQ(i,1);
                    
                end
            end
            
            i_vec = (eye(20)-Y_prime*C_cur)\(Y_prime*v_inv_vec+Y_prime*(C_Edp*Edp_vec+C_Eqp*Eqp_vec));
            
            for i = 1:10
                
                if i == 1
                    
                    i_mD(i,1) = i_vec(i);
                    i_mQ(i,1) = i_vec(i+10);
                    
                    i_md(i,1) = i_mD(i,1);
                    i_mq(i,1) = i_mQ(i,1);
                    
                    V_PSS1(i,1) = (omega(i,1)-omega_nom)*Kpss(i,1) - V_xPSS1(i,1);
                    V_PSS2(i,1) = 1/T2(i,1)*(T1(i,1)*V_PSS1(i,1) + V_xPSS2(i,1));
                    V_PSS(i,1) = 1/T4(i,1)*(T3(i,1)*V_PSS2(i,1) + V_xPSS3(i,1));
                    phi_d(i,1) = -xdp(i,1)*i_md(i,1) + Eqp(i,1);
                    phi_q(i,1) = -xqp(i,1)*i_mq(i,1) - Edp(i,1);
                    V_d(i,1) = -Ra(i,1)*i_md(i,1) - phi_q(i,1);
                    V_q(i,1) = -Ra(i,1)*i_mq(i,1) + phi_d(i,1);
                    V_T(i,1) = sqrt(V_d(i,1)^2+V_q(i,1)^2);
                    V_in(i,1) = V_ref(i,1) - V_T(i,1) + V_fb(i,1) - KF(i,1)/TF(i,1)*E_fd(i,1) + V_PSS(i,1);
                    T_e(i,1) = phi_d(i,1)*i_mq(i,1) - phi_q(i,1)*i_md(i,1);
                    %                 T_mech(i,1) = T_mech_eq(i,1);
                    
                    
                    
                    % x = [Eqp, Edp, delta, omega, V_f, V_a, V_fb, E_fd, V_xPSS1, V_xPSS2, V_xPSS3]
                    f_i = [];
                    
                    f_i(1,1) = 1/M(i,1) * (T_mech(i,1) - T_e(i,1) - D(i,1)*(omega(i,1)-omega_nom));
                    f_i(2,1) = 1/Td0p(i,1) * (-Eqp(i,1) - (xd(i,1)-xdp(i,1))*i_md(i,1) + E_fd(i,1));
                    %         f_i(3,1) = 1/Tq0p(i,1) * (-Edp(i,1) + (xq(i,1)-xqp(i,1))*i_mq(i,1));
                    f_i(3,1) = 1/TR(i,1) * (-V_f(i,1) + V_T(i,1));
                    f_i(4,1) = 1/TA(i,1)*(-V_a(i,1) + KA(i,1)*V_in(i,1));
                    f_i(5,1) = 1/TF(i,1)*(-V_fb(i,1) + KF(i,1)/TF(i,1)*E_fd(i,1));
                    f_i(6,1) = 1/TE(i,1) * (-E_fd(i,1) + V_a(i,1));
                    f_i(7,1) = V_PSS1(i,1)/TW(i,1);
                    f_i(8,1) = -V_PSS2(i,1) + V_PSS1(i,1);
                    f_i(9,1) = -V_PSS(i,1) + V_PSS2(i,1);
                    f_i(10,1) = 1/TM(i,1)*(-(T_mech(i,1)-T_mech_eq(i,1)) - 1/(RD(i,1)/sysdata_gen_Pmax(i,1)*100)*(omega(i,1)-omega_nom)); % T_mech
                    
                    f = [f;f_i];
                    
                    %             elseif i <= 10-N_inv
                elseif ~ismember(i,InverterBus_i)
                    
                    i_mD(i,1) = i_vec(i);
                    i_mQ(i,1) = i_vec(i+10);
                    
                    i_md(i,1) = cos(-delta(i,1)*omega_base_sys)*i_mD(i,1) - sin(-delta(i,1)*omega_base_sys)*i_mQ(i,1);
                    i_mq(i,1) = sin(-delta(i,1)*omega_base_sys)*i_mD(i,1) + cos(-delta(i,1)*omega_base_sys)*i_mQ(i,1);
                    
                    %         i_md(i,1) = I_d_eq(i,1);
                    %         i_mq(i,1) = I_q_eq(i,1);
                    
                    
                    V_PSS1(i,1) = (omega(i,1)-omega_nom)*Kpss(i,1) - V_xPSS1(i,1);
                    V_PSS2(i,1) = 1/T2(i,1)*(T1(i,1)*V_PSS1(i,1) + V_xPSS2(i,1));
                    V_PSS(i,1) = 1/T4(i,1)*(T3(i,1)*V_PSS2(i,1) + V_xPSS3(i,1));
                    phi_d(i,1) = -xdp(i,1)*i_md(i,1) + Eqp(i,1);
                    phi_q(i,1) = -xqp(i,1)*i_mq(i,1) - Edp(i,1);
                    V_d(i,1) = -Ra(i,1)*i_md(i,1) - phi_q(i,1);
                    V_q(i,1) = -Ra(i,1)*i_mq(i,1) + phi_d(i,1);
                    V_T(i,1) = sqrt(V_d(i,1)^2+V_q(i,1)^2);
                    V_in(i,1) = V_ref(i,1) - V_T(i,1) + V_fb(i,1) - KF(i,1)/TF(i,1)*E_fd(i,1) + V_PSS(i,1);
                    T_e(i,1) = phi_d(i,1)*i_mq(i,1) - phi_q(i,1)*i_md(i,1);
                    %                 T_mech(i,1) = T_mech_eq(i,1);
                    
                    f_i = [];
                    
                    f_i(1,1) = 1/M(i,1) * (T_mech(i,1) - T_e(i,1) - D(i,1)*(omega(i,1)-omega_nom));
                    f_i(2,1) = 1/Td0p(i,1) * (-Eqp(i,1) - (xd(i,1)-xdp(i,1))*i_md(i,1) + E_fd(i,1));
                    f_i(3,1) = 1/Tq0p(i,1) * (-Edp(i,1) + (xq(i,1)-xqp(i,1))*i_mq(i,1));
                    f_i(4,1) = 1/TR(i,1) * (-V_f(i,1) + V_T(i,1));
                    f_i(5,1) = 1/TA(i,1)*(-V_a(i,1) + KA(i,1)*V_in(i,1));
                    f_i(6,1) = 1/TF(i,1)*(-V_fb(i,1) + KF(i,1)/TF(i,1)*E_fd(i,1));
                    f_i(7,1) = 1/TE(i,1) * (-E_fd(i,1) + V_a(i,1));
                    f_i(8,1) = V_PSS1(i,1)/TW(i,1);
                    f_i(9,1) = -V_PSS2(i,1) + V_PSS1(i,1);
                    f_i(10,1) = -V_PSS(i,1) + V_PSS2(i,1);
                    f_i(11,1) = omega(i,1) - omega(1,1);
                    f_i(12,1) = 1/TM(i,1)*(-(T_mech(i,1)-T_mech_eq(i,1)) - 1/(RD(i,1)/sysdata_gen_Pmax(i,1)*100)*(omega(i,1)-omega_nom)); % T_mech
                    
                    f = [f;f_i];
                    
                else
                    
                    i_oD(i,1) = i_vec(i);
                    i_oQ(i,1) = i_vec(i+10);
                    
                    i_od(i,1) = cos(-delta(i,1)*omega_base_sys)*i_oD(i,1) - sin(-delta(i,1)*omega_base_sys)*i_oQ(i,1);
                    i_oq(i,1) = sin(-delta(i,1)*omega_base_sys)*i_oD(i,1) + cos(-delta(i,1)*omega_base_sys)*i_oQ(i,1);
                    
                    f_i = [];
                    
                    f_i(1,1) = xi(i,1)/(mu_v(i,1)^2)*v_id(i,1)*2*(V_nom(i,1)^2 - v_id(i,1)^2) - mu_v(i,1)*mu_i(i,1)/(3*Cvoc(i,1)*v_id(i,1))*(-v_id(i,1)*i_lq(i,1) - Q_st_ref(i,1)); % VOC voltage
                    f_i(2,1) = omega_nom - (mu_v(i,1)*mu_i(i,1)/(3*Cvoc(i,1)*v_id(i,1)^2)*(v_id(i,1)*i_ld(i,1) - P_st_ref(i,1)))/omega_base_sys - omega(1,1);
                    f_i(3,1) = (1/Lf(i,1)*(v_id(i,1) - v_od(i,1) - rf(i,1)*i_ld(i,1)) + omega_nom*i_lq(i,1))*omega_base_sys; %ild
                    f_i(4,1) = (1/Lf(i,1)*(v_iq(i,1) - v_oq(i,1) - rf(i,1)*i_lq(i,1)) - omega_nom*i_ld(i,1))*omega_base_sys; %ilq
                    f_i(5,1) = Rd(i,1)*f_i(3,1) + (1/Cf(i,1)*(i_ld(i,1) - i_od(i,1)) + omega_nom*v_oq(i,1) - omega_nom*Rd(i,1)*(i_lq(i,1) - i_oq(i,1)))*omega_base_sys; %vod
                    f_i(6,1) = Rd(i,1)*f_i(4,1) + (1/Cf(i,1)*(i_lq(i,1) - i_oq(i,1)) - omega_nom*v_od(i,1) + omega_nom*Rd(i,1)*(i_ld(i,1) - i_od(i,1)))*omega_base_sys; %voq
                    
                    f = [f;f_i];
                    
                end
                
            end
            
            f_minus = f;
            
            
            dfdx_i = (f_plus - f_minus)/(2*dx);
            dfdx = [dfdx,dfdx_i];
            
        end
        
        [eigvec_right,eigval_mat,eigvec_left] = eig(dfdx);
        eigval = diag(eigval_mat);
        
        eigval_all{i_ninv,:} = eigval';
        
        eigval_Rmax = max(real(eigval));
        
        eigval_Rmax_vec(i_ninv,ii) = eigval_Rmax;
        
        
    end
    
end
%%
figure
cmap = colormap(jet(nx));
for  i = 1:nx
    plot(inv_vec,eigval_Rmax_vec(:,i),'x','MarkerSize',10,'lINEWIDTH',2,'Color',cmap(i,:))
    hold on;
end
xlim([1 9])
set(gca,'fontsize',14,'fontname','Times New Roman','Units', 'Normalized', 'OuterPosition', [0, 0.04, 0.95, 0.96])
xrange = get(gca,'xlim');
%line(xrange,[0,0],'color','k','linestyle','--','MarkerSize',4,'Linewidth',2)
xlabel('No. of Inverter Buses')
ylabel('$\max (\mathrm{Re}(\lambda))$','interpreter','latex')
grid on
c = colorbar;
xticks([1 2 3 4 5 6 7 8 9])
xticklabels({'1','2','3','4','5','6','7','8','9'})
box off;

set(c,'XTick',linspace(0,1,2),'XTickLabel',{'Scale = 0.1','Scale = 10'})
% Edp_1 = Edp(1,1);
% save ('x_star_1_inv_eq.mat', 'x_star', 'T_mech_eq', 'Edp_1', 'V_ref_d');

% idx = find(real(eigval_all{i_ninv,:})'>0);
% figure;
% bar(abs(eigvec_right(:,idx(1))));
% xlabel('State No.','Interpreter','latex')
% ylabel('$|v_{kl}|$','Interpreter','latex')





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