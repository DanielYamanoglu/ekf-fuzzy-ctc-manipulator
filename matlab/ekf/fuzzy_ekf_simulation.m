%% FUZZY-EKF Integrated Simulation
% 2-DOF Robot Manipulator: Fuzzy Gain Scheduled CTC + EKF State Estimation
% Based on FINAL EKF and FINAL FUZZY implementations
% Date: 2026-01-08

clear; clc; close all;

% Output folder for figures
output_folder = fileparts(mfilename('fullpath'));
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

fprintf('================================================================================\n');
fprintf('           FUZZY-EKF INTEGRATED SIMULATION                                     \n');
fprintf('           2-DOF Robot Manipulator Control & Estimation                        \n');
fprintf('================================================================================\n\n');

%% ==================== SYSTEM PARAMETERS ====================
% Same as FINAL EKF
params.m1 = 1.0;   % Link 1 mass [kg]
params.m2 = 1.0;   % Link 2 mass [kg]
params.l1 = 0.1;   % Link 1 length [m]
params.l2 = 0.1;   % Link 2 length [m]
params.g = 9.81;   % Gravity [m/s²]

fprintf('System Parameters:\n');
fprintf('  m₁ = %.1f kg, m₂ = %.1f kg\n', params.m1, params.m2);
fprintf('  l₁ = %.1f m, l₂ = %.1f m\n', params.l1, params.l2);
fprintf('  g = %.2f m/s²\n\n', params.g);

%% ==================== SIMULATION SETTINGS ====================
T = 3.0;           % Simulation time [s]
dt = 0.01;         % Time step [s]
N = T/dt;          % Number of steps
t = (0:N-1)*dt;    % Time vector

% Reference trajectory
x_ref = [0.32; 0.09; 0; 0];  % [θ₁; θ₂; ω₁; ω₂]

fprintf('Simulation Settings:\n');
fprintf('  Duration: %.1f s, dt = %.3f s, Steps: %d\n', T, dt, N);
fprintf('  Reference: θ₁=%.2f rad, θ₂=%.2f rad\n\n', x_ref(1), x_ref(2));

%% ==================== EKF PARAMETERS ====================
% Optimized EKF parameters - Joint 2 omega2 smoother
P0 = diag([0.01, 0.01, 0.5, 0.5]);
Q = diag([1e-4, 1e-4, 0.02, 0.008]);   % omega2 reduced for smoother estimate
R = diag([2e-4, 2e-4]);

fprintf('EKF Parameters:\n');
fprintf('  P₀ diagonal: [%.2e, %.2e, %.2e, %.2e]\n', diag(P0));
fprintf('  Q diagonal:  [%.2e, %.2e, %.2e, %.2e]\n', diag(Q));
fprintf('  R diagonal:  [%.2e, %.2e]\n\n', diag(R));

%% ==================== INITIALIZE STATES ====================
% True system state (starts from zero)
x_true = [0; 0; 0; 0];

% EKF estimated state (with small initial error)
x_est = [0.02; -0.01; 0; 0];
P = P0;

% Storage arrays
x_true_hist = zeros(4, N);
x_est_hist = zeros(4, N);
y_meas_hist = zeros(2, N);
u_hist = zeros(2, N);
Kp_hist = zeros(2, N);
Kd_hist = zeros(2, N);
alpha_Kp_hist = zeros(2, N);
alpha_Kd_hist = zeros(2, N);
innovation_hist = zeros(2, N);
NIS_hist = zeros(1, N);
P_trace_hist = zeros(1, N);

%% ==================== MAIN SIMULATION LOOP ====================
fprintf('Running FUZZY-EKF simulation...\n');

for k = 1:N
    %% 1. FUZZY-CTC CONTROL (using estimated state)
    % Compute errors
    e_pos = x_est(1:2) - x_ref(1:2);
    e_vel = x_est(3:4) - x_ref(3:4);
    
    % Fuzzy Gain Scheduling
    [Kp, Kd, alpha_Kp, alpha_Kd] = fuzzy_gain_scheduler_local(e_pos, e_vel);
    
    % Desired acceleration
    a_desired = -Kp * e_pos - Kd * e_vel;
    
    % Robot dynamics matrices (from true state for CTC)
    [M, C_term, G] = robot_dynamics_local(x_true(1:2), x_true(3:4), params);
    
    % Computed Torque Control
    u = M * a_desired + C_term + G;
    
    % Torque saturation
    u_max = 15;
    u = max(min(u, u_max), -u_max);
    
    %% 2. TRUE SYSTEM DYNAMICS (RK4 Integration)
    k1 = system_dynamics_local(x_true, u, params);
    k2 = system_dynamics_local(x_true + dt/2*k1, u, params);
    k3 = system_dynamics_local(x_true + dt/2*k2, u, params);
    k4 = system_dynamics_local(x_true + dt*k3, u, params);
    x_true = x_true + dt/6*(k1 + 2*k2 + 2*k3 + k4);
    
    %% 3. GENERATE MEASUREMENT
    y_true = x_true(1:2);  % h(x) = [θ₁; θ₂]
    y_meas = y_true + sqrt(R(1,1))*randn(2,1);  % Add noise
    
    %% 4. EKF PREDICTION
    % State prediction (RK4)
    k1 = system_dynamics_local(x_est, u, params);
    k2 = system_dynamics_local(x_est + dt/2*k1, u, params);
    k3 = system_dynamics_local(x_est + dt/2*k2, u, params);
    k4 = system_dynamics_local(x_est + dt*k3, u, params);
    x_pred = x_est + dt/6*(k1 + 2*k2 + 2*k3 + k4);
    
    % Jacobian computation
    F = compute_jacobian_local(x_est, u, params);
    Phi = eye(4) + F*dt;
    
    % Covariance prediction
    P_pred = Phi*P*Phi' + Q;
    P_pred = 0.5*(P_pred + P_pred') + 1e-9*eye(4);
    
    %% 5. EKF UPDATE
    % Innovation
    y_pred = x_pred(1:2);
    innovation = y_meas - y_pred;
    
    % Measurement Jacobian
    H = [1 0 0 0; 0 1 0 0];
    
    % Innovation covariance
    S = H*P_pred*H' + R;
    S = 0.5*(S + S') + 1e-9*eye(2);
    
    % Kalman gain
    K = P_pred*H'/S;
    
    % State update
    x_est = x_pred + K*innovation;
    
    % Covariance update (Joseph form)
    I4 = eye(4);
    P = (I4 - K*H)*P_pred*(I4 - K*H)' + K*R*K';
    P = 0.5*(P + P') + 1e-9*eye(4);
    
    % NIS
    NIS = innovation'*(S\innovation);
    
    %% 6. STORE RESULTS
    x_true_hist(:,k) = x_true;
    x_est_hist(:,k) = x_est;
    y_meas_hist(:,k) = y_meas;
    u_hist(:,k) = u;
    Kp_hist(:,k) = diag(Kp);
    Kd_hist(:,k) = diag(Kd);
    alpha_Kp_hist(:,k) = alpha_Kp;
    alpha_Kd_hist(:,k) = alpha_Kd;
    innovation_hist(:,k) = innovation;
    NIS_hist(k) = NIS;
    P_trace_hist(k) = trace(P);
end

%% ==================== PERFORMANCE METRICS ====================
fprintf('\n--------------------------------------------------------------------------------\n');
fprintf('PERFORMANCE METRICS\n');
fprintf('--------------------------------------------------------------------------------\n');

% Estimation errors
est_error = vecnorm(x_est_hist - x_true_hist);
pos_error = vecnorm(x_est_hist(1:2,:) - x_true_hist(1:2,:));
vel_error = vecnorm(x_est_hist(3:4,:) - x_true_hist(3:4,:));

% Tracking errors (true state to reference)
tracking_error_1 = x_true_hist(1,:) - x_ref(1);
tracking_error_2 = x_true_hist(2,:) - x_ref(2);

fprintf('\n=== EKF STATE ESTIMATION ===\n');
fprintf('  Mean estimation error:  %.4f rad\n', mean(est_error));
fprintf('  Final estimation error: %.4f rad\n', est_error(end));
fprintf('  RMSE (position):        %.4f rad\n', rms(pos_error));
fprintf('  RMSE (velocity):        %.4f rad/s\n', rms(vel_error));
fprintf('  Mean NIS:               %.3f (target: ~2.0)\n', mean(NIS_hist));
fprintf('  NIS within bounds:      %.1f%%\n', sum(NIS_hist >= 0.1 & NIS_hist <= 5.99)/N*100);

fprintf('\n=== FUZZY-CTC TRACKING ===\n');
fprintf('  Settling time (θ₁):    %.2f s\n', find_settling_time(t, tracking_error_1, 0.02*x_ref(1)));
fprintf('  Settling time (θ₂):    %.2f s\n', find_settling_time(t, tracking_error_2, 0.02*x_ref(2)));
fprintf('  Final tracking error:   θ₁=%.4f, θ₂=%.4f rad\n', abs(tracking_error_1(end)), abs(tracking_error_2(end)));
fprintf('  Max torque:             τ₁=%.2f, τ₂=%.2f N·m\n', max(abs(u_hist(1,:))), max(abs(u_hist(2,:))));

fprintf('\n=== FUZZY GAIN ADAPTATION ===\n');
fprintf('  Kp range (Joint 1):    [%.1f, %.1f]\n', min(Kp_hist(1,:)), max(Kp_hist(1,:)));
fprintf('  Kp range (Joint 2):    [%.1f, %.1f]\n', min(Kp_hist(2,:)), max(Kp_hist(2,:)));
fprintf('  Kd range (Joint 1):    [%.1f, %.1f]\n', min(Kd_hist(1,:)), max(Kd_hist(1,:)));
fprintf('  Kd range (Joint 2):    [%.1f, %.1f]\n', min(Kd_hist(2,:)), max(Kd_hist(2,:)));
fprintf('  α_Kp variation:         %.1f%%\n', (max(max(alpha_Kp_hist)) - min(min(alpha_Kp_hist)))*50);

%% ==================== VISUALIZATION ====================
fprintf('\n--------------------------------------------------------------------------------\n');
fprintf('Generating plots...\n');

% Figure 1: State Estimation
fig1 = figure('Name', 'FUZZY-EKF State Estimation', 'Position', [50, 50, 1400, 900], 'Visible', 'off');

subplot(2,2,1);
plot(t, x_true_hist(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_est_hist(1,:), 'r--', 'LineWidth', 1.5);
plot(t, y_meas_hist(1,:), 'k.', 'MarkerSize', 3);
plot([0 T], [x_ref(1) x_ref(1)], 'g:', 'LineWidth', 2);
xlabel('Time [s]'); ylabel('\theta_1 [rad]');
title('Joint 1 Position');
legend('True', 'EKF Estimate', 'Measurement', 'Reference', 'Location', 'best');
grid on; ylim([-0.05 0.4]);

subplot(2,2,2);
plot(t, x_true_hist(2,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_est_hist(2,:), 'r--', 'LineWidth', 1.5);
plot(t, y_meas_hist(2,:), 'k.', 'MarkerSize', 3);
plot([0 T], [x_ref(2) x_ref(2)], 'g:', 'LineWidth', 2);
xlabel('Time [s]'); ylabel('\theta_2 [rad]');
title('Joint 2 Position');
legend('True', 'EKF Estimate', 'Measurement', 'Reference', 'Location', 'best');
grid on; ylim([-0.05 0.15]);

subplot(2,2,3);
plot(t, x_true_hist(3,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_est_hist(3,:), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('\omega_1 [rad/s]');
title('Joint 1 Velocity (Unmeasured)');
legend('True', 'EKF Estimate', 'Location', 'best');
grid on;

subplot(2,2,4);
plot(t, x_true_hist(4,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_est_hist(4,:), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('\omega_2 [rad/s]');
title('Joint 2 Velocity (Unmeasured)');
legend('True', 'EKF Estimate', 'Location', 'best');
grid on;

sgtitle('FUZZY-EKF: State Estimation Performance', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 2: Fuzzy Gain Adaptation
fig2 = figure('Name', 'Fuzzy Gain Adaptation', 'Position', [100, 100, 1400, 600], 'Visible', 'off');

subplot(2,3,1);
plot(t, Kp_hist(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, Kp_hist(2,:), 'r--', 'LineWidth', 1.5);
plot([0 T], [100 100], 'k:', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('K_p');
title('Proportional Gains');
legend('Joint 1', 'Joint 2', 'Nominal', 'Location', 'best');
grid on;

subplot(2,3,2);
plot(t, Kd_hist(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, Kd_hist(2,:), 'r--', 'LineWidth', 1.5);
plot([0 T], [50 50], 'k:', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('K_d');
title('Derivative Gains');
legend('Joint 1', 'Joint 2', 'Nominal', 'Location', 'best');
grid on;

subplot(2,3,3);
plot(t, alpha_Kp_hist(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, alpha_Kp_hist(2,:), 'r--', 'LineWidth', 1.5);
plot(t, alpha_Kd_hist(1,:), 'b:', 'LineWidth', 1.5);
plot(t, alpha_Kd_hist(2,:), 'r:', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('\alpha');
title('Fuzzy Scaling Factors');
legend('\alpha_{Kp,1}', '\alpha_{Kp,2}', '\alpha_{Kd,1}', '\alpha_{Kd,2}', 'Location', 'best');
grid on; ylim([0.4 2.1]);

subplot(2,3,4);
plot(t, u_hist(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, u_hist(2,:), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('\tau [N \cdot m]');
title('Control Torques');
legend('\tau_1', '\tau_2', 'Location', 'best');
grid on;

subplot(2,3,5);
plot(t, NIS_hist, 'b-', 'LineWidth', 1); hold on;
plot([0 T], [0.1 0.1], 'r--', 'LineWidth', 1);
plot([0 T], [5.99 5.99], 'r--', 'LineWidth', 1);
plot([0 T], [2.0 2.0], 'g--', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('NIS');
title('Normalized Innovation Squared');
legend('NIS', 'Bounds', '', 'Expected', 'Location', 'best');
grid on; ylim([0 10]);

subplot(2,3,6);
semilogy(t, P_trace_hist, 'b-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Trace(P)');
title('Covariance Trace');
grid on;

sgtitle('FUZZY-EKF: Fuzzy Gain Scheduling & EKF Metrics', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 3: Estimation & Tracking Errors
fig3 = figure('Name', 'Error Analysis', 'Position', [150, 150, 1200, 500], 'Visible', 'off');

subplot(1,3,1);
plot(t, pos_error, 'b-', 'LineWidth', 1.5); hold on;
plot(t, vel_error, 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Error [rad or rad/s]');
title('EKF Estimation Errors');
legend({'Position', 'Velocity'}, 'Location', 'best');
grid on;

subplot(1,3,2);
plot(t, tracking_error_1, 'b-', 'LineWidth', 1.5); hold on;
plot(t, tracking_error_2, 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Tracking Error [rad]');
title('Reference Tracking Errors');
legend('\theta_1 error', '\theta_2 error', 'Location', 'best');
grid on;

subplot(1,3,3);
plot(t, innovation_hist(1,:), 'b-', 'LineWidth', 1); hold on;
plot(t, innovation_hist(2,:), 'r--', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('Innovation [rad]');
title('EKF Innovations');
legend('\nu_1', '\nu_2', 'Location', 'best');
grid on;

sgtitle('FUZZY-EKF: Error Analysis', 'FontSize', 14, 'FontWeight', 'bold');

%% ==================== SAVE FIGURES ====================
fprintf('Saving figures to: %s\n', output_folder);
print(fig1, '-dpng', '-r150', fullfile(output_folder, 'FUZZY_EKF_State_Estimation.png'));
print(fig2, '-dpng', '-r150', fullfile(output_folder, 'FUZZY_EKF_Gain_Adaptation.png'));
print(fig3, '-dpng', '-r150', fullfile(output_folder, 'FUZZY_EKF_Error_Analysis.png'));
fprintf('All figures saved successfully.\n');

%% ==================== FINAL SUMMARY ====================
fprintf('\n================================================================================\n');
fprintf('SIMULATION COMPLETE\n');
fprintf('================================================================================\n');

% Overall grade
est_grade = 100 - min(100, mean(est_error)*1000);
nis_grade = 100 - min(100, abs(mean(NIS_hist) - 2.0)*50);
track_grade = 100 - min(100, (abs(tracking_error_1(end)) + abs(tracking_error_2(end)))*1000);
overall = (est_grade + nis_grade + track_grade) / 3;

fprintf('Performance Score: %.1f/100\n', overall);
fprintf('  - Estimation accuracy: %.1f/100\n', est_grade);
fprintf('  - Filter consistency:  %.1f/100\n', nis_grade);
fprintf('  - Tracking accuracy:   %.1f/100\n', track_grade);
fprintf('================================================================================\n');

%% ==================== LOCAL FUNCTIONS ====================

function x_dot = system_dynamics_local(x, u, params)
    theta1 = x(1); theta2 = x(2);
    thetad1 = x(3); thetad2 = x(4);
    
    m1 = params.m1; m2 = params.m2;
    l1 = params.l1; l2 = params.l2;
    g = params.g;
    
    % Mass matrix
    c2 = cos(theta2);
    M11 = (m1 + m2)*l1^2 + m2*l2^2 + 2*m2*l1*l2*c2;
    M12 = m2*l2^2 + m2*l1*l2*c2;
    M22 = m2*l2^2;
    M = [M11 M12; M12 M22];
    M = 0.5*(M + M') + 1e-6*eye(2);
    
    % Coriolis
    s2 = sin(theta2);
    C1 = -m2*l1*l2*s2*(2*thetad1*thetad2 + thetad2^2);
    C2 = m2*l1*l2*s2*thetad1^2;
    C = [C1; C2];
    
    % Gravity
    G1 = (m1 + m2)*l1*g*sin(theta1) + m2*l2*g*sin(theta1 + theta2);
    G2 = m2*l2*g*sin(theta1 + theta2);
    G = [G1; G2];
    
    theta_ddot = M \ (u - C - G);
    
    if any(isnan(theta_ddot)) || any(isinf(theta_ddot))
        theta_ddot = [0; 0];
    end
    
    x_dot = [thetad1; thetad2; theta_ddot];
end

function [M, C_term, G] = robot_dynamics_local(theta, thetad, params)
    m1 = params.m1; m2 = params.m2;
    l1 = params.l1; l2 = params.l2;
    g = params.g;
    
    theta1 = theta(1); theta2 = theta(2);
    thetad1 = thetad(1); thetad2 = thetad(2);
    
    c2 = cos(theta2);
    M11 = (m1 + m2)*l1^2 + m2*l2^2 + 2*m2*l1*l2*c2;
    M12 = m2*l2^2 + m2*l1*l2*c2;
    M22 = m2*l2^2;
    M = [M11 M12; M12 M22];
    M = 0.5*(M + M') + 1e-6*eye(2);
    
    s2 = sin(theta2);
    C1 = -m2*l1*l2*s2*(2*thetad1*thetad2 + thetad2^2);
    C2 = m2*l1*l2*s2*thetad1^2;
    C_term = [C1; C2];
    
    G1 = (m1 + m2)*l1*g*sin(theta1) + m2*l2*g*sin(theta1 + theta2);
    G2 = m2*l2*g*sin(theta1 + theta2);
    G = [G1; G2];
end

function F = compute_jacobian_local(x, u, params)
    n = 4; epsilon = 1e-7;
    F = zeros(n);
    
    for i = 1:n
        xp = x; xm = x;
        xp(i) = xp(i) + epsilon;
        xm(i) = xm(i) - epsilon;
        F(:,i) = (system_dynamics_local(xp, u, params) - ...
                  system_dynamics_local(xm, u, params)) / (2*epsilon);
    end
end

function [Kp, Kd, alpha_Kp, alpha_Kd] = fuzzy_gain_scheduler_local(e, e_dot)
    % Simplified fuzzy gain scheduler (based on fuzzy_gain_scheduler_optimized.m)
    
    persistent mf_centers rule_Kp rule_Kd out_centers initialized
    
    if isempty(initialized)
        mf_centers = [-1.0, -0.5, 0.0, 0.5, 1.0];
        out_centers = [0.625, 0.875, 1.25, 1.625, 1.875];
        
        rule_Kp = [5 5 4 3 2; 5 4 3 2 1; 4 3 3 3 4; 1 2 3 4 5; 2 3 4 5 5];
        rule_Kd = [2 3 4 5 5; 1 2 3 4 5; 1 2 3 2 1; 5 4 3 2 1; 5 5 4 3 2];
        
        initialized = true;
    end
    
    e_max = 0.5;
    e_dot_max = 2.0;
    % Joint-specific nominal gains: Joint 2 higher for faster settling
    Kp_nominal = [100, 160];  % Joint 2 increased
    Kd_nominal = [50, 70];    % Joint 2 increased
    
    alpha_Kp = zeros(2, 1);
    alpha_Kd = zeros(2, 1);
    
    for joint = 1:2
        e_norm = max(-1, min(1, e(joint) / e_max));
        ed_norm = max(-1, min(1, e_dot(joint) / e_dot_max));
        
        % Joint-specific sigma: Joint 2 narrower for faster response
        if joint == 1
            sigma = 0.35;
        else
            sigma = 0.25;  % Narrower Zero MF for Joint 2
        end
        mu_e = exp(-((e_norm - mf_centers).^2) / (2*sigma^2));
        mu_ed = exp(-((ed_norm - mf_centers).^2) / (2*sigma^2));
        
        mu_e = mu_e / (sum(mu_e) + 1e-10);
        mu_ed = mu_ed / (sum(mu_ed) + 1e-10);
        
        weighted_Kp = 0; weighted_Kd = 0;
        sum_w = 0;
        
        for i = 1:5
            for j = 1:5
                w = mu_e(i) * mu_ed(j);
                weighted_Kp = weighted_Kp + w * out_centers(rule_Kp(i,j));
                weighted_Kd = weighted_Kd + w * out_centers(rule_Kd(i,j));
                sum_w = sum_w + w;
            end
        end
        
        alpha_Kp(joint) = max(0.5, min(2.0, weighted_Kp / (sum_w + 1e-10)));
        alpha_Kd(joint) = max(0.5, min(2.0, weighted_Kd / (sum_w + 1e-10)));
    end
    
    Kp = diag([Kp_nominal(1) * alpha_Kp(1), Kp_nominal(2) * alpha_Kp(2)]);
    Kd = diag([Kd_nominal(1) * alpha_Kd(1), Kd_nominal(2) * alpha_Kd(2)]);
end

function ts = find_settling_time(t, error, threshold)
    % Find 2% settling time
    idx = find(abs(error) <= threshold, 1, 'first');
    if isempty(idx)
        ts = t(end);
    else
        % Check if it stays within threshold
        for k = idx:length(error)
            if abs(error(k)) > threshold
                idx = find(abs(error(k:end)) <= threshold, 1, 'first') + k - 1;
                if isempty(idx)
                    ts = t(end);
                    return;
                end
            end
        end
        ts = t(idx);
    end
end
