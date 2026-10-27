function [x_pred, P_pred] = ekf_predict(x, P, u, dt, Q, params)
% EKF prediction step

% RK4 integration
k1 = system_dynamics(x, u, params);
k2 = system_dynamics(x + dt/2*k1, u, params);
k3 = system_dynamics(x + dt/2*k2, u, params);
k4 = system_dynamics(x + dt*k3, u, params);
x_pred = x + dt/6*(k1 + 2*k2 + 2*k3 + k4);

% Validate prediction
if any(isnan(x_pred)) || any(isinf(x_pred))
    x_pred = x;
end

% Covariance prediction
F = compute_state_jacobian(x, u, params);
Phi = eye(4) + F*dt;
P_pred = Phi*P*Phi' + Q;
P_pred = 0.5*(P_pred + P_pred') + 1e-9*eye(4);

% Bound covariance diagonal
for i = 1:4
    if P_pred(i,i) > 10 || isnan(P_pred(i,i))
        P_pred(i,i) = 10;
    end
    if P_pred(i,i) < 1e-9
        P_pred(i,i) = 1e-9;
    end
end
end
