close all; clear; clc;

% this file shows the convergence of the Policy-Iterated trajectory to the
% exact trajectory

exact_data = load("Exact_state_data.mat");
PI_data = load("Policy_Iterated_state_data.mat");

iter = 8;

figure;
plot(1:120, exact_data.state1, LineWidth=1.5);
hold on
grid on
plot(1:120, PI_data.state1PI(:,8), '--', LineWidth=1.5);

figure;
hold on
for i = 2:8
    plot(1:120, PI_data.state1PI(:,i) - exact_data.state1, LineWidth=1.5);
end
grid on

figure;
plot(1:120, exact_data.state2, LineWidth=1.5);
hold on
grid on
plot(1:120, PI_data.state2PI(:,8), '--', LineWidth=1.5);

figure;
hold on
for i = 2:8
    plot(1:120, PI_data.state2PI(:,i) - exact_data.state2, LineWidth=1.5);
end
grid on