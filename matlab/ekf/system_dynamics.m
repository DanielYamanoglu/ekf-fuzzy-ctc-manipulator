function x_dot = system_dynamics(x, u, params)
% Robot manipulator dynamics: M(θ)θ̈ + C(θ,θ̇)θ̇ + G(θ) = τ

% Input validation
if any(isnan(x)) || any(isinf(x))
    x_dot = zeros(4,1);
    return;
end

theta1 = x(1); theta2 = x(2);
thetad1 = x(3); thetad2 = x(4);

m1 = params.m1; m2 = params.m2;
l1 = params.l1; l2 = params.l2;
g = params.g;

% Mass matrix
M11 = (m1 + m2)*l1^2 + m2*l2^2 + 2*m2*l1*l2*cos(theta2);
M12 = m2*l2^2 + m2*l1*l2*cos(theta2);
M21 = M12;
M22 = m2*l2^2;
M = [M11 M12; M21 M22];

% Symmetrize and regularize
M = 0.5*(M + M') + 1e-6*eye(2);

% Coriolis/centrifugal
C1 = -m2*l1*l2*sin(theta2)*(2*thetad1*thetad2 + thetad2^2);
C2 = m2*l1*l2*sin(theta2)*thetad1^2;
C = [C1; C2];

% Gravity
G1 = (m1 + m2)*l1*g*sin(theta1) + m2*l2*g*sin(theta1 + theta2);
G2 = m2*l2*g*sin(theta1 + theta2);
G = [G1; G2];

% Joint acceleration
theta_ddot = M \ (u - C - G);

% Output validation
if any(isnan(theta_ddot)) || any(isinf(theta_ddot))
    theta_ddot = [0; 0];
end

x_dot = [thetad1; thetad2; theta_ddot];
end
