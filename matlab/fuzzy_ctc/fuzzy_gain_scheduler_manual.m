function [Kp, Kd, alpha_Kp, alpha_Kd] = fuzzy_gain_scheduler_manual(e, e_dot)
%% FUZZY_GAIN_SCHEDULER_MANUAL - Manual Fuzzy Gain Scheduling (No Toolbox)
% ========================================
% Module: fuzzy_gain_scheduler_manual.m
% Description: Implements Mamdani FIS without Fuzzy Logic Toolbox
% Functionality: Fuzzification, rule evaluation, defuzzification
% Key Variables: Kp, Kd (gain matrices), alpha (scaling factors)
% Dependencies: None (standalone)
% Last Updated: December 2024
% ========================================
%
% This function implements a complete Mamdani Fuzzy Inference System
% for gain scheduling without requiring the Fuzzy Logic Toolbox.
%
% Inputs:
%   e     - position error [e1; e2] (rad), 2x1 vector
%   e_dot - velocity error [e_dot1; e_dot2] (rad/s), 2x1 vector
%
% Outputs:
%   Kp       - proportional gain matrix [2x2 diagonal]
%   Kd       - derivative gain matrix [2x2 diagonal]
%   alpha_Kp - Kp scaling factors [2x1] (optional)
%   alpha_Kd - Kd scaling factors [2x1] (optional)
%
% Usage:
%   [Kp, Kd] = fuzzy_gain_scheduler_manual(e, e_dot);
%   [Kp, Kd, alpha_Kp, alpha_Kd] = fuzzy_gain_scheduler_manual(e, e_dot);
%
% Author: Control Systems Engineering Team
% Project: 2-DOF Robot Manipulator - CTC + EKF + Fuzzy

%% ==================== PARAMETERS ====================
% Normalization parameters
e_max = 0.5;        % Max position error for normalization (rad)
e_dot_max = 2.0;    % Max velocity error for normalization (rad/s)

% Nominal gains
Kp_nominal = 100;   % Nominal Kp (N·m/rad)
Kd_nominal = 50;    % Nominal Kd (N·m·s/rad)

%% ==================== MEMBERSHIP FUNCTION PARAMETERS ====================
% Triangular MF parameters: [a, b, c] where a=left, b=peak, c=right

% Input MFs (same for e_norm and e_dot_norm)
% Labels: 1=NB, 2=NS, 3=Z, 4=PS, 5=PB
mf_in = [
    -1.0, -1.0, -0.5;  % NB (Negative Big)
    -1.0, -0.5,  0.0;  % NS (Negative Small)
    -0.5,  0.0,  0.5;  % Z  (Zero)
     0.0,  0.5,  1.0;  % PS (Positive Small)
     0.5,  1.0,  1.0   % PB (Positive Big)
];

% Output MFs (for alpha_Kp and alpha_Kd)
% Labels: 1=VL, 2=L, 3=M, 4=H, 5=VH
mf_out = [
    0.500, 0.500, 0.875;  % VL (Very Low)
    0.500, 0.875, 1.250;  % L  (Low)
    0.875, 1.250, 1.625;  % M  (Medium)
    1.250, 1.625, 2.000;  % H  (High)
    1.625, 2.000, 2.000   % VH (Very High)
];

%% ==================== RULE TABLES ====================
% Rule tables: rows = e MF (1-5), cols = e_dot MF (1-5)
% Values = output MF index (1=VL, 2=L, 3=M, 4=H, 5=VH)

% Kp scaling rules
rule_Kp = [
    5 5 4 3 2;  % e=NB: VH VH H  M  L
    5 4 3 2 1;  % e=NS: VH H  M  L  VL
    4 3 3 3 4;  % e=Z:  H  M  M  M  H
    1 2 3 4 5;  % e=PS: VL L  M  H  VH
    2 3 4 5 5   % e=PB: L  M  H  VH VH
];

% Kd scaling rules
rule_Kd = [
    2 3 4 5 5;  % e=NB: L  M  H  VH VH
    1 2 3 4 5;  % e=NS: VL L  M  H  VH
    1 2 3 2 1;  % e=Z:  VL L  M  L  VL
    5 4 3 2 1;  % e=PS: VH H  M  L  VL
    5 5 4 3 2   % e=PB: VH VH H  M  L
];

%% ==================== PROCESS EACH JOINT ====================
alpha_Kp = zeros(2, 1);
alpha_Kd = zeros(2, 1);

for joint = 1:2
    %% Step 1: Normalize inputs
    e_norm = max(-1, min(1, e(joint) / e_max));
    ed_norm = max(-1, min(1, e_dot(joint) / e_dot_max));
    
    %% Step 2: Fuzzification - compute membership degrees
    mu_e = zeros(5, 1);    % Membership degrees for error
    mu_ed = zeros(5, 1);   % Membership degrees for error derivative
    
    for i = 1:5
        mu_e(i) = trimf_eval(e_norm, mf_in(i,:));
        mu_ed(i) = trimf_eval(ed_norm, mf_in(i,:));
    end
    
    %% Step 3: Rule evaluation and aggregation
    % Discretize output universe for defuzzification
    n_points = 101;
    y_range = linspace(0.5, 2.0, n_points);
    
    % Aggregated output membership functions
    agg_Kp = zeros(1, n_points);
    agg_Kd = zeros(1, n_points);
    
    % Evaluate all 25 rules
    for i = 1:5  % e MF index
        for j = 1:5  % e_dot MF index
            % Rule firing strength (AND = min)
            alpha = min(mu_e(i), mu_ed(j));
            
            if alpha > 1e-10  % Only process active rules
                % Get output MF indices from rule tables
                out_idx_Kp = rule_Kp(i, j);
                out_idx_Kd = rule_Kd(i, j);
                
                % Implication and aggregation for Kp
                for k = 1:n_points
                    mu_out = trimf_eval(y_range(k), mf_out(out_idx_Kp,:));
                    clipped = min(alpha, mu_out);  % Implication (min)
                    agg_Kp(k) = max(agg_Kp(k), clipped);  % Aggregation (max)
                end
                
                % Implication and aggregation for Kd
                for k = 1:n_points
                    mu_out = trimf_eval(y_range(k), mf_out(out_idx_Kd,:));
                    clipped = min(alpha, mu_out);
                    agg_Kd(k) = max(agg_Kd(k), clipped);
                end
            end
        end
    end
    
    %% Step 4: Defuzzification (Centroid of Area)
    sum_agg_Kp = sum(agg_Kp);
    sum_agg_Kd = sum(agg_Kd);
    
    if sum_agg_Kp > 1e-10
        alpha_Kp(joint) = sum(y_range .* agg_Kp) / sum_agg_Kp;
    else
        alpha_Kp(joint) = 1.0;  % Default to nominal
    end
    
    if sum_agg_Kd > 1e-10
        alpha_Kd(joint) = sum(y_range .* agg_Kd) / sum_agg_Kd;
    else
        alpha_Kd(joint) = 1.0;  % Default to nominal
    end
end

%% ==================== COMPUTE FINAL GAINS ====================
Kp = diag([Kp_nominal * alpha_Kp(1), Kp_nominal * alpha_Kp(2)]);
Kd = diag([Kd_nominal * alpha_Kd(1), Kd_nominal * alpha_Kd(2)]);

end

%% ==================== HELPER FUNCTION ====================
function mu = trimf_eval(x, params)
%% TRIMF_EVAL - Evaluate triangular membership function
%
% Inputs:
%   x      - input value
%   params - [a, b, c] where a=left foot, b=peak, c=right foot
%
% Output:
%   mu - membership degree [0, 1]

a = params(1);
b = params(2);
c = params(3);

if x <= a
    mu = 0;
elseif x <= b
    if abs(b - a) < 1e-10
        mu = 1;  % Avoid division by zero (left shoulder)
    else
        mu = (x - a) / (b - a);
    end
elseif x <= c
    if abs(c - b) < 1e-10
        mu = 1;  % Avoid division by zero (right shoulder)
    else
        mu = (c - x) / (c - b);
    end
else
    mu = 0;
end

end
