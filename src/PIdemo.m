close all; clear; clc;

% This file implements the policy-iteration method for linear, time-varying
% discrete-time systems described in the paper.

% important values mentioned in the example in the paper
n = 2;
m = 1;
k0 = 0;
N = 120;
k = k0:N-1;

% system matrices
A = @(k) [1, 0.0025*k;
    -0.1*cos(0.3*k), 1 + (0.05^1.5)*sqrt(k)*sin(0.5*k)];
B = @(k) 0.05*[1; 1 - 1/(0.1*k + 3)];

% weight matrices
Q = @(k) (0.04*k + 2)*eye(n);
R = @(k) (5 - 0.02*k);
Q_N = 50*eye(n); % referred to as F in the paper


% value function matrices
iter = 8; % number of iterations for Policy Iteration

% initial guess for L matrix
L = zeros(1,n,N,iter+1);
V = zeros(n,n,N,iter);
x = zeros(n,1,N,iter+1);
% state-transition matrices
% Phi - 120 different matrices, where ith matrix is Phi(N,i)
Phi = zeros(n,n,N,iter);

% Policy Iteration Loop
for i = 1:iter 
    x(:,:,1,i) = [-2; 1]; % initial state in every iteration
    % compute the state transition matrices at the iteration
    % help matrices A_k,L
    % Phi - 120 different matrices, where ith matrix is Phi(N,i)
    Phi(:,:,N,i) = eye(n);
    for j = N-1:-1:k0+1
        Phi(:,:,j,i) = Phi(:,:,j+1,i)*(A(j) - B(j)*L(:,:,j,i));
    end
    % Compute the value function for the current policy at each time step
    V(:,:,N,i) = Q_N;
    % apply Lyapunov equation
    for j = N-1:-1:k0+1
        Abar = (A(j) - B(j)*L(:,:,j,i));
        V(:,:,j,i) = Abar'*V(:,:,j+1,i)*Abar + Q(j) + ...
                L(:,:,j,i)'*R(j)*L(:,:,j,i);
    end

    % Update the policy L based on the value function
    for j = N-1:-1:k0+1
        L(:,:,j,i+1) = inv(R(j) + B(j)'*V(:,:,j+1,i)*B(j))*B(j)'*V(:,:,j+1,i)*A(j); %#ok
    end

    % computing trajectory using current iteration matrices
    for j = k0+1:N-1
        x(:,:,j+1,i) = A(j)*x(:,:,j,i) - B(j)*L(:,:,j,i)*x(:,:,j,i);
    end
end

state1PI = zeros(N,1);
state2PI = zeros(N,1);
for i = 1:N
    for j = 1:iter
        state1PI(i,j) = x(1,:,i,j);
        state2PI(i,j) = x(2,:,i,j);
    end
end

figure;
hold on
for i = 1:iter
    plot(k0+1:N, state1PI(:,i), linewidth=1.5)
end
grid on
legend('iter1','iter2','iter3','iter4','iter5','iter6','iter7','iter8')

figure;
hold on
for i = 1:iter
    plot(k0+1:N, state2PI(:,i), linewidth=1.5)
end
grid on
legend('iter1','iter2','iter3','iter4','iter5','iter6','iter7','iter8')