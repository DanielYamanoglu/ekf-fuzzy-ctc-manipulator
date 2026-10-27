function F = compute_state_jacobian(x, u, params)
% State transition Jacobian F = ∂f/∂x
n = 4; epsilon = 1e-7;
F = zeros(n);
f0 = system_dynamics(x, u, params);

for i = 1:n
    xp = x; xm = x;
    xp(i) = xp(i) + epsilon;
    xm(i) = xm(i) - epsilon;
    F(:,i) = (system_dynamics(xp, u, params) - system_dynamics(xm, u, params))/(2*epsilon);
end
end
