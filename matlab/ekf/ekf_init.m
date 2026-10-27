function ekf = ekf_init(x0, P0, Q, R, params)
% Initialize Extended Kalman Filter

if isempty(P0), P0 = diag([0.03, 0.03, 1, 1]); end
if isempty(Q), Q = diag([1e-5, 1e-5, 0.02, 0.02]); end
if isempty(R), R = diag([1.5e-4, 1.5e-4]); end

ekf.x = x0(:);
ekf.P = 0.5*(P0 + P0');
ekf.Q = 0.5*(Q + Q');
ekf.R = 0.5*(R + R');
ekf.params = params;
ekf.initialized = true;
ekf.k = 0;
end
