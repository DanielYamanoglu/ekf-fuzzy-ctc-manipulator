%% Standard CTC Simulation (Baseline)
% 2-DOF Robot Manipulator: Computed Torque Control (Constant Gains)
% Ideal Sensor Feedback (No EKF)
% Date: 2026-01-08

clear; clc; close all;

% Output folder for figures (current folder)
output_folder = pwd;

fprintf('================================================================================\n');
fprintf('           STANDARD CTC SIMULATION (BASELINE)                                  \n');
fprintf('           2-DOF Robot Manipulator - Computed Torque Control (Fixed Gains)     \n');
fprintf('================================================================================\n\n');

%% ==================== SYSTEM PARAMETERS ====================
params.m1 = 1.0;   % Link 1 mass [kg]
params.m2 = 1.0;   % Link 2 mass [kg]
params.l1 = 0.1;   % Link 1 length [m]
params.l2 = 0.1;   % Link 2 length [m]
params.g = 9.81;   % Gravity [m/s²]

fprintf('System Parameters:\n');
fprintf('  m₁ = %.1f kg, m₂ = %.1f kg\n', params.m1, params.m2);
fprintf('  l₁ = %.1f m, l₂ = %.1f m\n', params.l1, params.l2);
fprintf('  g = %.2f m/s²\n\n', params.g);

%% ==================== CONTROL PARAMETERS ====================
% Fixed Gains (Baseline)
Kp_fixed = [100 0; 0 100];
Kd_fixed = [90 0; 0 70];  % Optimized values from Fuzzy study used as baseline

fprintf('Control Gains (Fixed):\n');
fprintf('  Kp = diag([%.1f, %.1f])\n', Kp_fixed(1,1), Kp_fixed(2,2));
fprintf('  Kd = diag([%.1f, %.1f])\n\n', Kd_fixed(1,1), Kd_fixed(2,2));

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

%% ==================== NOISE PARAMETERS ====================
% Realistic sensor noise (for authentic CTC comparison)
pos_noise_std = 0.04;    % Position sensor noise [rad] - realistic
vel_noise_std = 0.07;    % Velocity sensor noise [rad/s] - realistic
process_noise_std = 0.04;  % Process noise (model uncertainty)

%% ==================== INITIALIZE STATES ====================
x_true = [0; 0; 0; 0];  % True state [θ₁; θ₂; ω₁; ω₂]

% Storage arrays
x_true_hist = zeros(4, N);
x_meas_hist = zeros(4, N);
u_hist = zeros(2, N);

%% ==================== MAIN SIMULATION LOOP ====================
fprintf('Running Baseline CTC simulation...\n');

for k = 1:N
    %% 1. MEASUREMENT (Ideal Sensors)
    % Position: true + small noise
    theta_meas = x_true(1:2) + pos_noise_std*randn(2,1);
    % Velocity: true + small noise
    omega_meas = x_true(3:4) + vel_noise_std*randn(2,1);
    
    x_meas = [theta_meas; omega_meas];
    
    %% 2. CTC CONTROL (Fixed Gains)
    e_pos = x_meas(1:2) - x_ref(1:2);
    e_vel = x_meas(3:4) - x_ref(3:4);
    
    % Desired acceleration (PD control law)
    a_desired = -Kp_fixed * e_pos - Kd_fixed * e_vel;
    
    % Robot dynamics matrices
    [M, C_term, G] = robot_dynamics_local(x_meas(1:2), x_meas(3:4), params);
    
    % Computed Torque Control: τ = M*a_des + C + G
    u = M * a_desired + C_term + G;
    
    % Torque saturation
    u_max = 15;
    u = max(min(u, u_max), -u_max);
    
    %% 3. TRUE SYSTEM DYNAMICS (RK4 Integration)
    k1 = system_dynamics_local(x_true, u, params);
    k2 = system_dynamics_local(x_true + dt/2*k1, u, params);
    k3 = system_dynamics_local(x_true + dt/2*k2, u, params);
    k4 = system_dynamics_local(x_true + dt*k3, u, params);
    x_true = x_true + dt/6*(k1 + 2*k2 + 2*k3 + k4);
    
    % Add process noise
    x_true(3:4) = x_true(3:4) + process_noise_std*randn(2,1)*dt;
    
    %% 4. STORE RESULTS
    x_true_hist(:,k) = x_true;
    x_meas_hist(:,k) = x_meas;
    u_hist(:,k) = u;
end

%% ==================== PERFORMANCE METRICS ====================
fprintf('\n--------------------------------------------------------------------------------\n');
fprintf('PERFORMANCE METRICS\n');
fprintf('--------------------------------------------------------------------------------\n');

% Tracking errors
tracking_error_1 = x_true_hist(1,:) - x_ref(1);
tracking_error_2 = x_true_hist(2,:) - x_ref(2);

fprintf('\n=== CTC TRACKING ===\n');
fprintf('  Settling time (θ₁):    %.2f s\n', find_settling_time_local(t, tracking_error_1, 0.02*x_ref(1)));
fprintf('  Settling time (θ₂):    %.2f s\n', find_settling_time_local(t, tracking_error_2, 0.02*x_ref(2)));
fprintf('  Final tracking error:   θ₁=%.4f, θ₂=%.4f rad\n', abs(tracking_error_1(end)), abs(tracking_error_2(end)));
fprintf('  RMS Error (θ₁):        %.4f rad\n', rms(tracking_error_1));
fprintf('  RMS Error (θ₂):        %.4f rad\n', rms(tracking_error_2));
fprintf('  Max torque:             τ₁=%.2f, τ₂=%.2f N·m\n', max(abs(u_hist(1,:))), max(abs(u_hist(2,:))));

%% ==================== VISUALIZATION ====================
fprintf('\n--------------------------------------------------------------------------------\n');
fprintf('Generating plots...\n');

% Figure 1: State Tracking Performance (WITHOUT EKF ESTIMATE)
fig1 = figure('Name', 'Standard CTC State Tracking', 'Position', [50, 50, 1400, 900], 'Visible', 'off');

subplot(2,2,1);
plot(t, x_true_hist(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_meas_hist(1,:), 'k.', 'MarkerSize', 2);
plot([0 T], [x_ref(1) x_ref(1)], 'g:', 'LineWidth', 2);
xlabel('Time [s]'); ylabel('\theta_1 [rad]');
title('Joint 1 Position');
legend({'True State', 'Measurement', 'Reference'}, 'Location', 'best');
grid on; ylim([-0.05 0.4]);

subplot(2,2,2);
plot(t, x_true_hist(2,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_meas_hist(2,:), 'k.', 'MarkerSize', 2);
plot([0 T], [x_ref(2) x_ref(2)], 'g:', 'LineWidth', 2);
xlabel('Time [s]'); ylabel('\theta_2 [rad]');
title('Joint 2 Position');
legend({'True State', 'Measurement', 'Reference'}, 'Location', 'best');
grid on; ylim([-0.05 0.15]);

subplot(2,2,3);
plot(t, x_true_hist(3,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_meas_hist(3,:), 'r--', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('\omega_1 [rad/s]');
title('Joint 1 Angular Velocity (Unmeasured)');
legend({'True State', 'Noisy Sensor (Ideal)'}, 'Location', 'best');
grid on;

subplot(2,2,4);
plot(t, x_true_hist(4,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_meas_hist(4,:), 'r--', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('\omega_2 [rad/s]');
title('Joint 2 Angular Velocity (Unmeasured)');
legend({'True State', 'Noisy Sensor (Ideal)'}, 'Location', 'best');
grid on;

sgtitle('Standard CTC Control (No EKF)', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 2: Control Torques & Errors
fig2 = figure('Name', 'CTC Control Efforts', 'Position', [100, 100, 1200, 600], 'Visible', 'off');

subplot(2,2,1);
plot(t, u_hist(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, u_hist(2,:), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('\tau [N \cdot m]');
title('Control Torques');
legend('\tau_1', '\tau_2', 'Location', 'best');
grid on;

subplot(2,2,2);
plot(t, tracking_error_1, 'b-', 'LineWidth', 1.5); hold on;
plot(t, tracking_error_2, 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Error [rad]');
title('Tracking Errors');
legend('\theta_1', '\theta_2', 'Location', 'best');
grid on;

subplot(2,2,3);
% Torque Rate
du = diff(u_hist, 1, 2) / dt;
plot(t(2:end), du(1,:), 'b-', 'LineWidth', 1); hold on;
plot(t(2:end), du(2,:), 'r--', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('d\tau/dt');
title('Torque Rate');
grid on;

subplot(2,2,4);
% Energy
energy = cumsum(sum(u_hist.^2, 1)) * dt;
plot(t, energy, 'k-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Energy [J]');
title('Ref. Cumulative Energy');
grid on;

sgtitle('Standard CTC Performance Metrics', 'FontSize', 14, 'FontWeight', 'bold');

%% ==================== SAVE FIGURES ====================
fprintf('Saving figures to: %s\n', output_folder);
print(fig1, '-dpng', '-r150', fullfile(output_folder, 'CTC_State_Response.png'));
print(fig2, '-dpng', '-r150', fullfile(output_folder, 'CTC_Control_Analysis.png'));
fprintf('All figures saved successfully.\n');

%% ==================== LOCAL FUNCTIONS ====================

function x_dot = system_dynamics_local(x, u, params)
    theta1 = x(1); theta2 = x(2);
    thetad1 = x(3); thetad2 = x(4);
    
    m1 = params.m1; m2 = params.m2;
    l1 = params.l1; l2 = params.l2;
    g = params.g;
    
    c2 = cos(theta2);
    M11 = (m1 + m2)*l1^2 + m2*l2^2 + 2*m2*l1*l2*c2;
    M12 = m2*l2^2 + m2*l1*l2*c2;
    M22 = m2*l2^2;
    M = [M11 M12; M12 M22];
    M = 0.5*(M + M') + 1e-6*eye(2);
    
    s2 = sin(theta2);
    C1 = -m2*l1*l2*s2*(2*thetad1*thetad2 + thetad2^2);
    C2 = m2*l1*l2*s2*thetad1^2;
    C = [C1; C2];
    
    G1 = (m1 + m2)*l1*g*sin(theta1) + m2*l2*g*sin(theta1 + theta2);
    G2 = m2*l2*g*sin(theta1 + theta2);
    Gvec = [G1; G2];
    
    theta_ddot = M \ (u - C - Gvec);
    
    if any(isnan(theta_ddot)) || any(isinf(theta_ddot))
        theta_ddot = [0; 0];
    end
    
    x_dot = [thetad1; thetad2; theta_ddot];
end

function [M, C_vec, G_vec] = robot_dynamics_local(theta, theta_dot, params)
    theta1 = theta(1); theta2 = theta(2);
    thetad1 = theta_dot(1); thetad2 = theta_dot(2);
    
    m1 = params.m1; m2 = params.m2;
    l1 = params.l1; l2 = params.l2;
    g = params.g;
    
    c2 = cos(theta2);
    M11 = (m1 + m2)*l1^2 + m2*l2^2 + 2*m2*l1*l2*c2;
    M12 = m2*l2^2 + m2*l1*l2*c2;
    M22 = m2*l2^2;
    M = [M11 M12; M12 M22];
    M = 0.5*(M + M') + 1e-6*eye(2);
    
    s2 = sin(theta2);
    C1 = -m2*l1*l2*s2*(2*thetad1*thetad2 + thetad2^2);
    C2 = m2*l1*l2*s2*thetad1^2;
    C_vec = [C1; C2];
    
    G1 = (m1 + m2)*l1*g*sin(theta1) + m2*l2*g*sin(theta1 + theta2);
    G2 = m2*l2*g*sin(theta1 + theta2);
    G_vec = [G1; G2];
end

function ts = find_settling_time_local(t, error, threshold)
    idx = find(abs(error) <= threshold, 1, 'first');
    if isempty(idx)
        ts = t(end);
    else
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
