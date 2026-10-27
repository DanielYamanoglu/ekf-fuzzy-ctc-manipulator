%% TEST_OPTIMIZATION_PERFORMANCE - Optimization Comparison Test
% ========================================
% Module: test_optimization_performance.m
% Description: Compares performance of original and optimized fuzzy scheduler
% Last Updated: December 2024
% ========================================

clear; clc;
fprintf('================================================================================\n');
fprintf('     FUZZY GAIN SCHEDULER OPTIMIZATION PERFORMANCE TEST                        \n');
fprintf('================================================================================\n\n');

%% ==================== TEST PARAMETERS ====================
n_iterations = 1000;  % Number of test loop iterations
n_test_points = 50;   % Number of test points per loop

%% ==================== GENERATE RANDOM TEST POINTS ====================
rng(42);  % For reproducibility

% Test inputs: random error and velocity error
test_e = (rand(2, n_test_points) - 0.5) * 1.0;       % [-0.5, 0.5] rad
test_e_dot = (rand(2, n_test_points) - 0.5) * 4.0;   % [-2.0, 2.0] rad/s

%% ==================== ORIGINAL VERSION TEST ====================
fprintf('Testing original fuzzy_gain_scheduler_manual...\n');

tic;
for iter = 1:n_iterations
    for i = 1:n_test_points
        [Kp_orig, Kd_orig, ~, ~] = fuzzy_gain_scheduler_manual(test_e(:,i), test_e_dot(:,i));
    end
end
time_original = toc;

fprintf('  Total time: %.3f seconds\n', time_original);
fprintf('  Average/call: %.3f ms\n', time_original / (n_iterations * n_test_points) * 1000);

%% ==================== OPTIMIZED VERSION TEST ====================
fprintf('\nTesting fuzzy_gain_scheduler_optimized...\n');

tic;
for iter = 1:n_iterations
    for i = 1:n_test_points
        [Kp_opt, Kd_opt, ~, ~] = fuzzy_gain_scheduler_optimized(test_e(:,i), test_e_dot(:,i));
    end
end
time_optimized = toc;

fprintf('  Total time: %.3f seconds\n', time_optimized);
fprintf('  Average/call: %.3f ms\n', time_optimized / (n_iterations * n_test_points) * 1000);

%% ==================== PERFORMANCE COMPARISON ====================
speedup = time_original / time_optimized;
improvement_pct = (1 - time_optimized / time_original) * 100;

fprintf('\n================================================================================\n');
fprintf('                         PERFORMANCE RESULTS                                   \n');
fprintf('================================================================================\n\n');
fprintf('+-------------------------+--------------+---------------+\n');
fprintf('| Metric                  |  Original    |  Optimized    |\n');
fprintf('+-------------------------+--------------+---------------+\n');
fprintf('| Total time (s)          | %10.3f   | %10.3f    |\n', time_original, time_optimized);
fprintf('| Per call (ms)           | %10.4f   | %10.4f    |\n', ...
    time_original / (n_iterations * n_test_points) * 1000, ...
    time_optimized / (n_iterations * n_test_points) * 1000);
fprintf('+-------------------------+--------------+---------------+\n');
fprintf('| Speedup                 |          %.2fx               |\n', speedup);
fprintf('| Improvement             |         %+.1f%%              |\n', improvement_pct);
fprintf('+-------------------------+-------------------------------+\n');

%% ==================== ACCURACY COMPARISON ====================
fprintf('\n================================================================================\n');
fprintf('                         ACCURACY COMPARISON                                   \n');
fprintf('================================================================================\n\n');

% Compare at different test points
test_cases = {
    [0.3; -0.2], [1.0; -0.5], 'Large error, reverse vel';
    [0.0; 0.0], [0.0; 0.0], 'Zero error and velocity';
    [-0.4; 0.3], [-1.5; 1.2], 'Mixed signs';
    [0.1; -0.1], [0.5; -0.3], 'Small error';
    [0.5; 0.5], [2.0; 2.0], 'Maximum values';
};

fprintf('+----------------------------+------------+------------+------------+\n');
fprintf('| Test Case                  | Orig aKp   | Opt aKp    | Diff (%%)   |\n');
fprintf('+----------------------------+------------+------------+------------+\n');

total_error = 0;
n_cases = size(test_cases, 1);

for i = 1:n_cases
    e = test_cases{i, 1};
    e_dot = test_cases{i, 2};
    desc = test_cases{i, 3};
    
    [~, ~, alpha_orig, ~] = fuzzy_gain_scheduler_manual(e, e_dot);
    [~, ~, alpha_opt, ~] = fuzzy_gain_scheduler_optimized(e, e_dot);
    
    % Compare for Joint 1
    diff_pct = abs(alpha_orig(1) - alpha_opt(1)) / alpha_orig(1) * 100;
    total_error = total_error + diff_pct;
    
    fprintf('| %-26s | %10.3f | %10.3f | %9.2f%% |\n', ...
        desc, alpha_orig(1), alpha_opt(1), diff_pct);
end

fprintf('+----------------------------+------------+------------+------------+\n');
fprintf('| Average difference         |            %.2f%%                     |\n', total_error / n_cases);
fprintf('+----------------------------+-------------------------------------+\n');

%% ==================== CONCLUSION ====================
fprintf('\n================================================================================\n');
if speedup >= 2
    fprintf(' [OK] OPTIMIZATION SUCCESSFUL: %.1fx speedup achieved!\n', speedup);
else
    fprintf(' [!]  Limited speedup: %.1fx\n', speedup);
end

if total_error / n_cases < 5
    fprintf(' [OK] ACCURACY PRESERVED: Average difference < 5%%\n');
else
    fprintf(' [!]  Warning: Average difference > 5%% - calibration needed\n');
end
fprintf('================================================================================\n');
