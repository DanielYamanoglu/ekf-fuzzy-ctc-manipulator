% % PID Controller Performance - Report Format clear;
clc;
close all;

m1 = 1;
m2 = 0.8;
l1 = 0.5;
l2 = 0.4;
g = 9.81;
dt = 0.001;
t = 0 : dt : 3;
N = length(t);
Kp1 = 100;
Ki1 = 20;
Kd1 = 15;
Kp2 = 80;
Ki2 = 15;
Kd2 = 12;
ref1 = 0.32 * ones(1, N);
ref2 = 0.09 * ones(1, N);
x = [0; 0; 0; 0];
hist = zeros(4, N);
tau = zeros(2, N);
err = zeros(2, N);
ie = [0; 0];

for
  k = 1 : N hist( :, k) = x;
e1 = ref1(k) - x(1);
e2 = ref2(k) - x(2);
ie = ie + [e1; e2] * dt;
ie = max(-1, min(1, ie));
err( :, k) = [e1; e2];
t1 = Kp1 * e1 + Ki1 * ie(1) - Kd1 * x(3);
t2 = Kp2 * e2 + Ki2 * ie(2) - Kd2 * x(4);
t1 = max(-50, min(50, t1));
t2 = max(-50, min(50, t2));
tau( :, k) = [t1; t2];
M11 = (m1 + m2) * l1 ^ 2 + m2 * l2 ^ 2 + 2 * m2 * l1 * l2 * cos(x(2));
M = [
  M11, m2 *l2 ^ 2 + m2 * l1 * l2 *cos(x(2));
  m2 * l2 ^ 2 + m2 * l1 * l2 * cos(x(2)), m2 *l2 ^ 2
];
h = m2 * l1 * l2 * sin(x(2));
C = [ -h * x(4), -h *(x(3) + x(4)); h * x(3), 0 ];
G = [(m1 + m2) * l1 * g * sin(x(1)) + m2 * l2 * g * sin(x(1) + x(2));
     m2 * l2 * g * sin(x(1) + x(2))];
a = M\([t1; t2] - C * x(3 : 4) - G);
x(3 : 4) = x(3 : 4) + a * dt;
x(1 : 2) = x(1 : 2) + x(3 : 4) * dt;
end

% Add noise for measurement visualization
meas1 = hist(1,:) + 0.01*randn(1,N);
meas2 = hist(2, :) + 0.01 * randn(1, N);

figure('Position', [100 100 900 700]);
sgtitle('PID Control: State Estimation Performance', 'FontSize', 12,
        'FontWeight', 'bold');

% Joint 1 Position subplot(2, 2, 1);
plot(t, hist(1, :), 'b-', 'LineWidth', 1.2);
hold on;
plot(t, meas1, 'r.', 'MarkerSize', 1);
plot(t, ref1, 'g--', 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('\theta_1 [rad]');
title('Joint 1 Position');
legend('True', 'Measurement', 'Reference', 'Location', 'best');
grid on;
xlim([0 3]);

% Joint 2 Position subplot(2, 2, 2);
plot(t, hist(2, :), 'b-', 'LineWidth', 1.2);
hold on;
plot(t, meas2, 'r.', 'MarkerSize', 1);
plot(t, ref2, 'g--', 'LineWidth', 1.5);
xlabel('Time [s]');
ylabel('\theta_2 [rad]');
title('Joint 2 Position');
legend('True', 'Measurement', 'Reference', 'Location', 'best');
grid on;
xlim([0 3]);

% Joint 1 Velocity subplot(2, 2, 3);
plot(t, hist(3, :), 'b-', 'LineWidth', 1.2);
hold on;
noisy_vel1 = hist(3, :) + 0.05 * randn(1, N);
plot(t, noisy_vel1, 'r--', 'LineWidth', 0.8);
xlabel('Time [s]');
ylabel('\omega_1 [rad/s]');
title('Joint 1 Velocity (Unmeasured)');
legend('True', 'Noisy Sensor (Ideal)', 'Location', 'best');
grid on;
xlim([0 3]);

% Joint 2 Velocity subplot(2, 2, 4);
plot(t, hist(4, :), 'b-', 'LineWidth', 1.2);
hold on;
noisy_vel2 = hist(4, :) + 0.05 * randn(1, N);
plot(t, noisy_vel2, 'r--', 'LineWidth', 0.8);
xlabel('Time [s]');
ylabel('\omega_2 [rad/s]');
title('Joint 2 Velocity (Unmeasured)');
legend('True', 'Noisy Sensor (Ideal)', 'Location', 'best');
grid on;
xlim([0 3]);

saveas(gcf, 'pid_state_estimation.png');
disp('Saved: pid_state_estimation.png');