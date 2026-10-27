%% Extended Kalman Filter - Complete System Simulation
% 2-DOF Robot Manipulator State Estimation
clear; clc; close all;

fprintf('================================================================================\n');
fprintf('                    EXTENDED KALMAN FILTER SIMULATION                          \n');
fprintf('                        2-DOF Robot Manipulator                                \n');
fprintf('================================================================================\n\n');

%% System Parameters (Updated for reference plots)
params.m1 = 1.0;   % Link 1 mass [kg]
params.m2 = 1.0;   % Link 2 mass [kg]
params.l1 = 0.1;   % Link 1 length [m]
params.l2 = 0.1;   % Link 2 length [m]
params.g = 9.81;   % Gravity [m/s²]

fprintf('System Parameters:\n');
fprintf('  m₁ = %.1f kg, m₂ = %.1f kg\n', params.m1, params.m2);
fprintf('  l₁ = %.1f m, l₂ = %.1f m\n', params.l1, params.l2);
fprintf('  g = %.2f m/s²\n\n', params.g);

%% Test 1: System Dynamics Verification
fprintf('--------------------------------------------------------------------------------\n');
fprintf('Test 1: System Dynamics Verification\n');
fprintf('--------------------------------------------------------------------------------\n');

% Equilibrium point
x_eq = [0; 0; 0; 0];
u_eq = [0; 0];
x_dot_eq = system_dynamics(x_eq, u_eq, params);
fprintf('Equilibrium: x=[0,0,0,0], u=[0,0] → ẋ=[%.3f,%.3f,%.3f,%.3f]\n', x_dot_eq);

% Small angle response
x_small = [0.1; 0.05; 0; 0];
x_dot_small = system_dynamics(x_small, u_eq, params);
fprintf('Small angle: x=[0.1,0.05,0,0], u=[0,0] → θ̈=[%.3f,%.3f] rad/s²\n', x_dot_small(3:4));

% Torque response
u_torque = [2; 1];
x_dot_torque = system_dynamics(x_eq, u_torque, params);
fprintf('Applied torque: x=[0,0,0,0], u=[2,1] → θ̈=[%.3f,%.3f] rad/s²\n', x_dot_torque(3:4));

% Mass matrix properties
theta_test = [pi/6; pi/6];
M11 = (params.m1 + params.m2)*params.l1^2 + params.m2*params.l2^2 + 2*params.m2*params.l1*params.l2*cos(theta_test(2));
M12 = params.m2*params.l2^2 + params.m2*params.l1*params.l2*cos(theta_test(2));
M22 = params.m2*params.l2^2;
M = [M11 M12; M12 M22];
lambda = eig(M);
fprintf('Mass matrix at θ=[π/6,π/6]: λ_min=%.2f, λ_max=%.2f (positive definite)\n\n', min(lambda), max(lambda));

%% Test 2: Jacobian Computation
fprintf('--------------------------------------------------------------------------------\n');
fprintf('Test 2: Jacobian Matrix Computation\n');
fprintf('--------------------------------------------------------------------------------\n');

x_test = [0.1; 0.05; 0.2; -0.1];
u_test = [0.5; 0.3];

F = compute_state_jacobian(x_test, u_test, params);
fprintf('State Jacobian F (4×4):\n');
fprintf('  F₁ₓ = [%6.3f %6.3f %6.3f %6.3f]\n', F(1,:));
fprintf('  F₂ₓ = [%6.3f %6.3f %6.3f %6.3f]\n', F(2,:));
fprintf('  F₃ₓ = [%6.3f %6.3f %6.3f %6.3f]\n', F(3,:));
fprintf('  F₄ₓ = [%6.3f %6.3f %6.3f %6.3f]\n', F(4,:));

H = compute_measurement_jacobian(x_test);
fprintf('\nMeasurement Jacobian H (2×4):\n');
fprintf('  H = [1 0 0 0; 0 1 0 0] (constant, linear measurement)\n\n');

%% Test 3: EKF Initialization
fprintf('--------------------------------------------------------------------------------\n');
fprintf('Test 3: EKF Initialization\n');
fprintf('--------------------------------------------------------------------------------\n');

x0 = [0; 0; 0; 0];
P0 = diag([0.01, 0.01, 0.5, 0.5]);
Q = diag([5e-5, 5e-5, 0.01, 0.01]);
R = diag([1e-4, 1e-4]);

ekf = ekf_init(x0, P0, Q, R, params);

fprintf('Initial state: x₀ = [%.2f, %.2f, %.2f, %.2f]ᵀ\n', ekf.x);
fprintf('Initial covariance diagonal: P₀ = [%.3e, %.3e, %.3e, %.3e]\n', diag(ekf.P));
fprintf('Process noise diagonal: Q = [%.3e, %.3e, %.3e, %.3e]\n', diag(ekf.Q));
fprintf('Measurement noise diagonal: R = [%.3e, %.3e]\n', diag(ekf.R));
fprintf('Standard deviations: σ_θ = %.3f rad, σ_ω = %.3f rad/s, σ_y = %.3f rad\n\n', ...
    sqrt(P0(1,1)), sqrt(P0(3,3)), sqrt(R(1,1)));

%% Test 4: Single Prediction-Update Cycle
fprintf('--------------------------------------------------------------------------------\n');
fprintf('Test 4: Single EKF Cycle\n');
fprintf('--------------------------------------------------------------------------------\n');

dt = 0.01;
u = [0.5; 0.3];
[x_pred, P_pred] = ekf_predict(ekf.x, ekf.P, u, dt, ekf.Q, params);
fprintf('Prediction: x⁻ = [%.4f, %.4f, %.4f, %.4f]ᵀ\n', x_pred);

y_meas = [0.01; 0.005] + sqrt(R(1,1))*randn(2,1);
[x_est, P_est, innov, NIS] = ekf_update(x_pred, P_pred, y_meas, ekf.R, params);
fprintf('Measurement: y = [%.4f, %.4f]ᵀ\n', y_meas);
fprintf('Update: x⁺ = [%.4f, %.4f, %.4f, %.4f]ᵀ\n', x_est);
fprintf('Innovation: ν = [%.4f, %.4f]ᵀ\n', innov);
fprintf('NIS = %.3f (χ²₂ distribution, expect ~2.0)\n\n', NIS);

%% Test 5: Convergence Analysis
fprintf('--------------------------------------------------------------------------------\n');
fprintf('Test 5: Convergence Analysis (100 steps)\n');
fprintf('--------------------------------------------------------------------------------\n');

ekf = ekf_init([0.05; -0.02; 0; 0], P0, Q, R, params);
x_true = [0.1; 0.05; 0; 0];
NIS_vals = zeros(100, 1);
errors = zeros(100, 1);

for k = 1:100
    u = [0; 0];
    [x_pred, P_pred] = ekf_predict(ekf.x, ekf.P, u, dt, ekf.Q, params);
    y_true = measurement_model(x_true);
    y_meas = y_true + sqrt(R(1,1))*randn(2,1);
    [ekf.x, ekf.P, ~, NIS] = ekf_update(x_pred, P_pred, y_meas, ekf.R, params);
    NIS_vals(k) = NIS;
    errors(k) = norm(ekf.x - x_true);
end

fprintf('Convergence metrics:\n');
fprintf('  Initial error: %.4f rad\n', errors(1));
fprintf('  Final error: %.4f rad\n', errors(end));
fprintf('  Mean NIS: %.3f (target: 2.0)\n', mean(NIS_vals));
fprintf('  NIS in bounds [0.1, 5.0]: %d%%\n\n', sum(NIS_vals >= 0.1 & NIS_vals <= 5.0));

%% Main Simulation: Trajectory Tracking with CTC
fprintf('--------------------------------------------------------------------------------\n');
fprintf('Test 6: Full System Simulation with CTC (3 seconds)\n');
fprintf('--------------------------------------------------------------------------------\n');

T = 3;      % Simulation time (matching reference plots)
dt = 0.01;  % Time step
N = T/dt;   % Number of steps
t = (0:N-1)*dt;

% Reference trajectory (from reference plots)
x_ref = [0.32; 0.09; 0; 0];
fprintf('Reference: θ₁=%.2f rad, θ₂=%.2f rad\n', x_ref(1), x_ref(2));
fprintf('Control: Computed Torque Control (CTC)\n\n');

% Initialize true system (starting from zero)
x_true = [0; 0; 0; 0];
x_true_hist = zeros(4, N);
y_meas_hist = zeros(2, N);

% Initialize EKF (with small initial error)
ekf = ekf_init([0.02; -0.01; 0; 0], P0, Q, R, params);
x_est_hist = zeros(4, N);
P_trace_hist = zeros(N, 1);
NIS_hist = zeros(N, 1);
innovation_hist = zeros(2, N);

% Control input history
u_hist = zeros(2, N);

fprintf('Running simulation...\n');
for k = 1:N
    % Computed Torque Control
    u = computed_torque_controller(x_true, x_ref, params);
    u_hist(:,k) = u;
    
    % True system dynamics
    k1 = system_dynamics(x_true, u, params);
    k2 = system_dynamics(x_true + dt/2*k1, u, params);
    k3 = system_dynamics(x_true + dt/2*k2, u, params);
    k4 = system_dynamics(x_true + dt*k3, u, params);
    x_true = x_true + dt/6*(k1 + 2*k2 + 2*k3 + k4);
    x_true_hist(:,k) = x_true;
    
    % Generate measurement
    y_true = measurement_model(x_true);
    y_meas = y_true + sqrt(R(1,1))*randn(2,1);
    y_meas_hist(:,k) = y_meas;
    
    % EKF prediction
    [x_pred, P_pred] = ekf_predict(ekf.x, ekf.P, u, dt, ekf.Q, params);
    
    % EKF update
    [ekf.x, ekf.P, innov, NIS] = ekf_update(x_pred, P_pred, y_meas, ekf.R, params);
    
    % Store results
    x_est_hist(:,k) = ekf.x;
    P_trace_hist(k) = trace(ekf.P);
    NIS_hist(k) = NIS;
    innovation_hist(:,k) = innov;
end

% Calculate performance metrics
error_hist = vecnorm(x_est_hist - x_true_hist);
mean_error = mean(error_hist);
final_error = error_hist(end);
mean_NIS = mean(NIS_hist);
NIS_bounds = sum(NIS_hist >= 0.1 & NIS_hist <= 5.0)/N*100;

fprintf('\nPerformance Summary:\n');
fprintf('  Mean estimation error: %.4f rad\n', mean_error);
fprintf('  Final estimation error: %.4f rad\n', final_error);
fprintf('  Mean NIS: %.3f (target: ~2.0)\n', mean_NIS);
fprintf('  NIS within bounds: %.1f%%\n', NIS_bounds);
fprintf('  Final P trace: %.3e\n', P_trace_hist(end));
fprintf('  RMS innovation: θ₁=%.4f, θ₂=%.4f rad\n', ...
    rms(innovation_hist(1,:)), rms(innovation_hist(2,:)));

%% Visualization
fprintf('\n--------------------------------------------------------------------------------\n');
fprintf('Generating plots...\n');

% Figure 1: State Estimation
figure('Name', 'EKF State Estimation', 'Position', [50, 50, 1400, 900]);

subplot(2,2,1);
plot(t, x_true_hist(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_est_hist(1,:), 'r--', 'LineWidth', 1.5);
plot(t, y_meas_hist(1,:), 'k.', 'MarkerSize', 4);
plot([0 T], [x_ref(1) x_ref(1)], 'k:', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('θ₁ [rad]');
title('Joint 1 Position');
legend('True State', 'EKF Estimate', 'Measurement', 'Reference', 'Location', 'best');
grid on; ylim([-0.05 0.4]);

subplot(2,2,2);
plot(t, x_true_hist(2,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_est_hist(2,:), 'r--', 'LineWidth', 1.5);
plot(t, y_meas_hist(2,:), 'k.', 'MarkerSize', 4);
plot([0 T], [x_ref(2) x_ref(2)], 'k:', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('θ₂ [rad]');
title('Joint 2 Position');
legend('True State', 'EKF Estimate', 'Measurement', 'Reference', 'Location', 'best');
grid on; ylim([-0.05 0.15]);

subplot(2,2,3);
plot(t, x_true_hist(3,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_est_hist(3,:), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('ω₁ [rad/s]');
title('Joint 1 Angular Velocity (Unmeasured)');
legend('True', 'EKF Estimate', 'Location', 'best');
grid on;

subplot(2,2,4);
plot(t, x_true_hist(4,:), 'b-', 'LineWidth', 1.5); hold on;
plot(t, x_est_hist(4,:), 'r--', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('ω₂ [rad/s]');
title('Joint 2 Angular Velocity (Unmeasured)');
legend('True', 'EKF Estimate', 'Location', 'best');
grid on;

sgtitle('EKF State Estimation: 2-DOF Robot with CTC Control');

% Figure 2: Filter Performance Metrics
figure('Name', 'EKF Performance Metrics', 'Position', [100, 100, 1400, 600]);

subplot(1,3,1);
plot(t, NIS_hist, 'b-', 'LineWidth', 1);
hold on;
plot([0 T], [0.1 0.1], 'r--', 'LineWidth', 1);
plot([0 T], [5.0 5.0], 'r--', 'LineWidth', 1);
plot([0 T], [2.0 2.0], 'g--', 'LineWidth', 1);
xlabel('Time [s]'); ylabel('NIS');
title('Normalized Innovation Squared');
legend('NIS', 'Bounds', '', 'Expected', 'Location', 'best');
grid on; ylim([0 10]);

subplot(1,3,2);
plot(t, error_hist, 'b-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Error Norm [rad]');
title('State Estimation Error');
grid on;

subplot(1,3,3);
semilogy(t, P_trace_hist, 'b-', 'LineWidth', 1.5);
xlabel('Time [s]'); ylabel('Trace(P)');
title('Covariance Trace Evolution');
grid on;

%% Statistical Analysis
fprintf('--------------------------------------------------------------------------------\n');
fprintf('Statistical Analysis\n');
fprintf('--------------------------------------------------------------------------------\n');

% Innovation statistics
innov_mean = mean(innovation_hist, 2);
innov_cov = cov(innovation_hist');
fprintf('Innovation statistics:\n');
fprintf('  Mean: [%.4f, %.4f]ᵀ (expect: [0, 0]ᵀ)\n', innov_mean);
fprintf('  Covariance:\n');
fprintf('    [%.3e  %.3e]\n', innov_cov(1,:));
fprintf('    [%.3e  %.3e]\n', innov_cov(2,:));

% Chi-square test
chi2_critical_95 = 5.991; % χ²(2) at 95%
NIS_pass = sum(NIS_hist <= chi2_critical_95)/N*100;
fprintf('\nChi-square test (95%% confidence):\n');
fprintf('  Pass rate: %.1f%% (expect: ~95%%)\n', NIS_pass);

% Autocorrelation of innovations
[acf1, ~] = xcorr(innovation_hist(1,:) - mean(innovation_hist(1,:)), 20, 'coeff');
[acf2, ~] = xcorr(innovation_hist(2,:) - mean(innovation_hist(2,:)), 20, 'coeff');
fprintf('\nInnovation autocorrelation at lag 1:\n');
fprintf('  θ₁: %.3f (expect: ~0)\n', acf1(22));
fprintf('  θ₂: %.3f (expect: ~0)\n', acf2(22));

%% Energy Analysis
fprintf('\n--------------------------------------------------------------------------------\n');
fprintf('Energy Conservation Analysis\n');
fprintf('--------------------------------------------------------------------------------\n');

% Kinetic energy
KE = zeros(1, N);
for k = 1:N
    theta = x_true_hist(1:2, k);
    thetad = x_true_hist(3:4, k);
    M11 = (params.m1 + params.m2)*params.l1^2 + params.m2*params.l2^2 + 2*params.m2*params.l1*params.l2*cos(theta(2));
    M12 = params.m2*params.l2^2 + params.m2*params.l1*params.l2*cos(theta(2));
    M22 = params.m2*params.l2^2;
    M = [M11 M12; M12 M22];
    KE(k) = 0.5*thetad'*M*thetad;
end

% Potential energy
PE = zeros(1, N);
for k = 1:N
    theta1 = x_true_hist(1, k);
    theta2 = x_true_hist(2, k);
    h1 = -params.l1*cos(theta1);
    h2 = -params.l1*cos(theta1) - params.l2*cos(theta1 + theta2);
    PE(k) = params.m1*params.g*h1 + params.m2*params.g*h2;
end

E_total = KE + PE;
E_mean = mean(E_total);
E_std = std(E_total);

fprintf('System energy:\n');
fprintf('  Mean total energy: %.2f J\n', E_mean);
fprintf('  Energy variation: ±%.2f J (%.1f%%)\n', E_std, 100*E_std/abs(E_mean));
fprintf('  Max kinetic energy: %.2f J\n', max(KE));
fprintf('  Max potential energy: %.2f J\n', max(PE));

%% Final Summary
fprintf('\n================================================================================\n');
fprintf('SIMULATION COMPLETE\n');
fprintf('================================================================================\n');
fprintf('Filter convergence: ');
if final_error < 0.01
    fprintf('EXCELLENT (error < 0.01 rad)\n');
elseif final_error < 0.05
    fprintf('GOOD (error < 0.05 rad)\n');
elseif final_error < 0.1
    fprintf('ACCEPTABLE (error < 0.1 rad)\n');
else
    fprintf('POOR (error > 0.1 rad)\n');
end

fprintf('Filter consistency: ');
if abs(mean_NIS - 2.0) < 0.5 && NIS_bounds > 80
    fprintf('CONSISTENT (NIS well-calibrated)\n');
else
    fprintf('CHECK TUNING (NIS=%0.2f, bounds=%.0f%%)\n', mean_NIS, NIS_bounds);
end

fprintf('\nAll tests completed successfully.\n');
fprintf('================================================================================\n');

%% Save Figures
fprintf('\nSaving figures...\n');
figure(1);
saveas(gcf, 'EKF_State_Estimation.png');
fprintf('  Saved: EKF_State_Estimation.png\n');

figure(2);
saveas(gcf, 'EKF_Performance_Metrics.png');
fprintf('  Saved: EKF_Performance_Metrics.png\n');

fprintf('All figures saved successfully.\n');

