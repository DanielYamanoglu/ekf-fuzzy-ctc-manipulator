function [M, C_term, G] = robot_dynamics_common(theta, thetad, params)
%% ROBOT_DYNAMICS_COMMON - Common Robot Dynamic Model Computations
% ========================================
% Module: robot_dynamics_common.m
% Description: Computes M(θ), C(θ,θ̇), G(θ) matrices
% Purpose: Eliminate code duplication (DRY principle)
% Last Updated: December 2024
% ========================================
%
% This function computes for a 2-DOF robot manipulator:
%   - Mass matrix M(θ)
%   - Coriolis/centrifugal term C(θ,θ̇)·θ̇
%   - Gravity vector G(θ)
%
% Inputs:
%   theta  - joint positions [theta1; theta2] (rad)
%   thetad - joint velocities [thetad1; thetad2] (rad/s)
%   params - system parameters struct (m1, m2, l1, l2, g)
%
% Outputs:
%   M      - 2x2 mass matrix
%   C_term - 2x1 Coriolis/centrifugal vector
%   G      - 2x1 gravity vector

%% ==================== INPUT VALIDATION ====================
if nargin < 3
    error('robot_dynamics_common: Insufficient arguments. theta, thetad, and params required.');
end

if length(theta) < 2 || length(thetad) < 2
    error('robot_dynamics_common: theta and thetad must be 2-element vectors.');
end

%% ==================== EXTRACT PARAMETERS ====================
m1 = params.m1;
m2 = params.m2;
l1 = params.l1;
l2 = params.l2;
g = params.g;

theta1 = theta(1);
theta2 = theta(2);
thetad1 = thetad(1);
thetad2 = thetad(2);

%% ==================== MASS MATRIX M(θ) ====================
% M = [M11 M12; M21 M22]
c2 = cos(theta2);
M11 = (m1 + m2)*l1^2 + m2*l2^2 + 2*m2*l1*l2*c2;
M12 = m2*l2^2 + m2*l1*l2*c2;
M21 = M12;
M22 = m2*l2^2;

M = [M11 M12; M21 M22];

% Symmetrize and regularize (for numerical stability)
M = 0.5*(M + M') + 1e-6*eye(2);

%% ==================== CORIOLIS/CENTRIFUGAL TERM C(θ,θ̇)·θ̇ ====================
s2 = sin(theta2);
h = m2*l1*l2*s2;

C1 = -h*(2*thetad1*thetad2 + thetad2^2);
C2 = h*thetad1^2;

C_term = [C1; C2];

%% ==================== GRAVITY VECTOR G(θ) ====================
s1 = sin(theta1);
s12 = sin(theta1 + theta2);

G1 = (m1 + m2)*l1*g*s1 + m2*l2*g*s12;
G2 = m2*l2*g*s12;

G = [G1; G2];

end
