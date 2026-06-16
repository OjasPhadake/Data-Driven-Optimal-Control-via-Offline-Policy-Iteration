close all; clear; clc;

% this file simulates the system described in the paper

% system parameters
n = 2; % states
m = 1; % inputs
k0 = 0;
N = 120;
k = k0:N-1;

% system matrices
A = @(k) [1, 0.0025*k;
    -0.1*cos(0.3*k), 1 + (0.05^1.5)*sqrt(k)*sin(0.5*k)];
B = @(k) 0.05*[1; 1 - 1/(0.1*k + 3)];

% weighing matrices
Q = @(k) (0.04*k + 2)*eye(n);
R = @(k) (5 - 0.02*k);
Q_N = 50*eye(n); % referred to as F in the paper

% solving the Algebraic Riccati equation to obtain P matrices and hence
% optimal control inputs
P = zeros(n, n, 120);
P(:, :, 120) = Q_N;

% obtaining all the P matrices
for i = N-1:-1:k0+1
    P(:,:,i) = Q(i) + (A(i)')*P(:,:,i+1)*A(i) - ...
    (A(i)')*P(:,:,i+1)*B(i)*inv(R(i) + (B(i)')*P(:,:,i+1)*B(i))*(B(i)')*P(:,:,i+1)*A(i); %#ok
end

% obtaining the gain matrices L
L = zeros(m,n,N);
for i = k0+1:N-1
    L(:,:,i) = inv(R(i) + (B(i)')*P(:,:,i+1)*B(i))*(B(i)')*P(:,:,i+1)*A(i);
end

% getting the optimal trajectory
x = zeros(n,1,N);
x(:,:,1) = [-2; 1]; % initial condition as given in the paper
u = zeros(N,1);

for i = k0+1:N-1
    u(i) = -L(:,:,i)*x(:,:,i); % optimal control input
    x(:,:,i+1) = A(i)*x(:,:,i) + B(i)*u(i); % control law
end

state1 = zeros(120,1);
state2 = zeros(120,1);

for i = k0+1:N
    state1(i) = x(1,:,i);
    state2(i) = x(2,:,i);
end

% plotting out the simulated state progression
figure;
plot(1:N, state1, linewidth=1.5)
grid on
hold on
title('Progression of State 1')

figure;
plot(1:N, state2, linewidth=1.5)
grid on
hold on
title('Progression of State 2')
