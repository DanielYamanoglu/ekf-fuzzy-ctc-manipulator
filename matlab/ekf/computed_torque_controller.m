function u = computed_torque_controller(x, x_ref, params)
% Computed Torque Control (CTC) - Eliminates oscillations
% Control Law: u = M(θ)·a_d + C(θ,θ̇)·θ̇ + G(θ)
%
% Inputs:
%   x - current state [theta1; theta2; thetad1; thetad2]
%   x_ref - reference state [theta1_ref; theta2_ref; 0; 0]
%   params - system parameters (m1, m2, l1, l2, g)
% Output:
%   u - control torques [tau1; tau2]

theta = x(1:2);
thetad = x(3:4);
theta_ref = x_ref(1:2);

m1 = params.m1;
m2 = params.m2;
l1 = params.l1;
l2 = params.l2;
g = params.g;

% Position and velocity errors
e_pos = theta - theta_ref;
e_vel = thetad;  % Reference velocity is zero

% PD gains (tuned for smooth convergence)
Kp = diag([100, 100]);
Kd = diag([50, 50]);

% Desired acceleration
a_desired = -Kp * e_pos - Kd * e_vel;

% Mass Matrix M(θ)
theta1 = theta(1);
theta2 = theta(2);
M11 = (m1 + m2)*l1^2 + m2*l2^2 + 2*m2*l1*l2*cos(theta2);
M12 = m2*l2^2 + m2*l1*l2*cos(theta2);
M21 = M12;
M22 = m2*l2^2;
M = [M11 M12; M21 M22];
M = 0.5*(M + M') + 1e-6*eye(2);  % Symmetrize and regularize

% Coriolis/Centrifugal Terms C(θ,θ̇)·θ̇
thetad1 = thetad(1);
thetad2 = thetad(2);
C1 = -m2*l1*l2*sin(theta2)*(2*thetad1*thetad2 + thetad2^2);
C2 = m2*l1*l2*sin(theta2)*thetad1^2;
C_term = [C1; C2];

% Gravity G(θ)
G1 = (m1 + m2)*l1*g*sin(theta1) + m2*l2*g*sin(theta1 + theta2);
G2 = m2*l2*g*sin(theta1 + theta2);
G = [G1; G2];

% Computed Torque Control: u = M(θ)·a_d + C(θ,θ̇)·θ̇ + G(θ)
u = M * a_desired + C_term + G;

% Torque saturation
u_max = 15;
u = max(min(u, u_max), -u_max);

end
