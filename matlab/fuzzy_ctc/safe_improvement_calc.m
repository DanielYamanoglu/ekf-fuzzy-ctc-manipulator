function [improvement, is_valid] = safe_improvement_calc(baseline, optimized, mode)
%% SAFE_IMPROVEMENT_CALC - Safe Percentage Improvement Calculation
% ========================================
% Module: safe_improvement_calc.m
% Description: Safely handles division by zero and edge cases
% Last Updated: December 2024
% ========================================
%
% Inputs:
%   baseline  - reference value
%   optimized - optimized value
%   mode      - 'lower_better' or 'higher_better'
%
% Outputs:
%   improvement - improvement percentage
%   is_valid    - whether the calculation is valid

if nargin < 3
    mode = 'lower_better';  % Default: lower value is better
end

% Default values
improvement = 0;
is_valid = true;

% Edge case check
if isnan(baseline) || isnan(optimized) || isinf(baseline) || isinf(optimized)
    is_valid = false;
    return;
end

% Near-zero value check
eps_threshold = 1e-10;

if abs(baseline) < eps_threshold
    if abs(optimized) < eps_threshold
        % Both near zero - equal performance
        improvement = 0;
        return;
    else
        % Baseline is zero but optimized is not
        % Cannot show infinite improvement/degradation
        is_valid = false;
        return;
    end
end

% Normal calculation
if strcmp(mode, 'lower_better')
    % Lower value is better (error, settling time, etc.)
    improvement = ((baseline - optimized) / abs(baseline)) * 100;
elseif strcmp(mode, 'higher_better')
    % Higher value is better (accuracy, efficiency, etc.)
    improvement = ((optimized - baseline) / abs(baseline)) * 100;
else
    error('Invalid mode. Use "lower_better" or "higher_better".');
end

% Clamp extreme values (for display)
improvement = max(-999.9, min(999.9, improvement));

end
