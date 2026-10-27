function [x_est, P_est, innovation, NIS] = ekf_update(x_pred, P_pred, y_meas, R, params)
% EKF measurement update

y_pred = measurement_model(x_pred);
innovation = y_meas - y_pred;

% Validate innovation
if any(isnan(innovation)) || any(isinf(innovation))
    x_est = x_pred;
    P_est = P_pred;
    innovation = zeros(2,1);
    NIS = 0;
    return;
end

H = compute_measurement_jacobian(x_pred);
S = H*P_pred*H' + R;
S = 0.5*(S + S') + 1e-9*eye(2);  % Regularize

K = P_pred*H'/S;

x_est = x_pred + K*innovation;

% Validate estimate
if any(isnan(x_est)) || any(isinf(x_est))
    x_est = x_pred;
    K = zeros(4,2);
end

% Joseph form covariance update
I = eye(4);
P_est = (I - K*H)*P_pred*(I - K*H)' + K*R*K';
P_est = 0.5*(P_est + P_est') + 1e-9*eye(4);

% Normalized Innovation Squared
try
    NIS = innovation'*(S\innovation);
    if isnan(NIS) || isinf(NIS)
        NIS = 0;
    end
catch
    NIS = 0;
end
end
